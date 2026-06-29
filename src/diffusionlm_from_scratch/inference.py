"""Inference for the masked (absorbing-state) text-diffusion DiT.

A lightweight, training-free home for the confidence-based parallel sampler
(RESEARCH.md section 6), the post-hoc ``refine`` pass, and the high-level
:class:`DiffusionLM` pipeline. Depends only on the model + scheduler (no
``accelerate``/``datasets``), so importing it for generation stays cheap.
"""

import math

import torch

from .model import DiT
from .scheduler import AbsorbingScheduler

DEFAULT_REPO = "tchauffi/diffusionlm-from-scratch"


@torch.no_grad()
def sample(model, scheduler, tokenizer, seq_len, n=4, steps=64, temperature=1.0,
          reveal_schedule="cosine", order="confidence", return_ids=False,
          corrector_frac=0.0, corrector_every=2, corrector_mode="random",
          confidence_threshold=None):
    """Confidence-based parallel sampler (RESEARCH.md section 6).

    Start from the all-``[MASK]`` prior at t=1 and walk t down to 0, committing
    tokens until the whole sequence is filled (the strict absorbing process: once
    a position is unmasked it stays). Runs on whatever device the model lives on.

    ``reveal_schedule``: how many tokens to commit per step -- ``"cosine"`` (few
    early, more late) or ``"linear"``. ``order``: which positions to commit --
    ``"confidence"`` (surest first; front-to-back here), ``"confidence_weighted"``
    (confidence-biased but spread across the sequence), or ``"random"``.

    ``confidence_threshold`` (e.g. 0.9): adaptive sampler -- commit *every* masked
    position whose max-prob exceeds the threshold (>=1 per step for progress),
    ignoring ``reveal_schedule``/``order``. The model's confidence is well-
    calibrated (max-prob ~= accuracy), so the threshold roughly means "only commit
    positions at least this likely to be correct" -- a quality/adaptivity knob.
    Note: it does NOT speed this model up. Confidence is front-loaded (a position
    is only confident once its left context exists), so early steps clear the
    threshold for ~1 token and it runs ~seq_len steps. The forward pass is
    conditioned on the current masked fraction so ``t`` tracks the actual masking.
    Use ``steps >= seq_len`` so it can finish.

    ``corrector_frac > 0`` turns this into a **predictor-corrector** sampler: every
    ``corrector_every`` steps, a ``corrector_frac`` of already-committed tokens is
    re-masked and re-predicted in place, so the model can fix earlier commits
    *during* generation (not only via :func:`refine` afterwards). Costs one extra
    forward per corrector step. ``corrector_frac=0`` (default) is the plain strict
    sampler. ``corrector_mode`` selects *which* commits to revisit:

    - ``"random"``: re-mask a random fraction. The stable default.
    - ``"low_confidence"``: re-mask the *lowest-confidence* commits -- the ones most
      likely wrong (a position committed early, before its context existed). This is
      the principled target, but re-prediction can drift toward locally-frequent
      tokens, so it is sampled (not argmaxed) and the corrected confidence is
      tracked so a position is not re-masked forever. (Re-masking *high*-confidence
      commits instead collapses into repetition -- "Mrs Mrs Mrs" -- so don't.)

    Returns decoded strings, or the (n, seq_len) id tensor if ``return_ids``.
    """
    model.eval()
    mask_id = scheduler.mask_id
    device = next(model.parameters()).device
    x = torch.full((n, seq_len), mask_id, device=device, dtype=torch.long)
    # Confidence at which each position was committed (0 = still masked). Lets the
    # low_confidence corrector target the shakiest commits.
    commit_conf = torch.zeros(n, seq_len, device=device)

    for i in range(steps):
        still = x == mask_id
        if not still.any() and corrector_frac == 0:
            break                                               # done; nothing to correct

        # Condition on the t whose *training* masked-fraction matches the CURRENT
        # masked fraction, i.e. t = sigma^{-1}(masked_frac). The model only ever saw
        # (x_t, t) pairs where masked_frac == sigma(t); a linear ts[i] (or, for the
        # cosine schedule, using the masked fraction directly as t) feeds it OOD
        # pairs -- it is told "nearly clean" while the input is still mostly masked.
        m = still.float().mean(dim=1)                           # (n,) masked fraction
        t_used = scheduler.t_from_mask_prob(m)
        logits = model(x, t_used)
        probs = (logits / temperature).softmax(-1)

        # Token: sample per position (not argmax) so the n samples diverge from the
        # identical all-[MASK] start and we don't collapse onto the single most
        # frequent token.
        pred = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1).reshape(n, -1)
        # Rank positions by peakedness (max prob), not the sampled token's prob: at
        # high masking the model predicts "." at a flat ~0.1 everywhere, and ranking
        # by sampled-token prob lets those flat positions commit early and cascade.
        conf = probs.max(-1).values                             # (n, L)

        # PREDICTOR: reveal more tokens (only while some remain masked).
        if still.any():
            conf_m = conf.masked_fill(~still, -1.0)             # only masked positions

            if confidence_threshold is not None:
                # Adaptive: commit every masked position the model is sure enough
                # about (>=1 for progress). Calibration makes the threshold mean
                # roughly "commit positions at least this likely to be correct".
                for b in range(n):
                    commit = conf_m[b] >= confidence_threshold
                    if not commit.any():
                        commit = torch.zeros_like(commit)
                        commit[conf_m[b].argmax()] = True       # at least one
                    x[b, commit] = pred[b, commit]
                    commit_conf[b, commit] = conf[b, commit]
            else:
                # Schedule-based reveal: target UNMASKED count this step. "cosine"
                # (MaskGIT-style) reveals few early -- little context, shouldn't freeze
                # guesses -- and accelerates later; "linear" is a constant rate.
                r = (i + 1) / steps
                if reveal_schedule == "cosine":
                    target_unmasked = (1.0 - math.cos(math.pi / 2 * r)) * seq_len
                else:  # "linear"
                    target_unmasked = r * seq_len
                for b in range(n):
                    k = max(1, int(round(target_unmasked - int((~still[b]).sum()))))
                    k = min(k, int(still[b].sum()))
                    if k <= 0:
                        continue
                    if order == "confidence":
                        # Commit the surest positions. On this model confidence is
                        # concentrated at the (predictable) story opening, so this
                        # decodes roughly front-to-back.
                        idx = conf_m[b].topk(k).indices
                    elif order == "confidence_weighted":
                        # Sample k positions weighted by confidence (conf is -1 at
                        # non-masked, so clamp keeps only masked positions eligible).
                        # Confidence bias but spread across the whole sequence.
                        idx = torch.multinomial(conf_m[b].clamp(min=0.0), k, replacement=False)
                    else:  # "random": commit random masked positions -> fills spread
                        cand = still[b].nonzero().squeeze(-1)
                        idx = cand[torch.randperm(cand.numel(), device=x.device)[:k]]
                    x[b, idx] = pred[b, idx]
                    commit_conf[b, idx] = conf[b, idx]

        # CORRECTOR: re-mask a fraction of committed tokens and re-predict them with
        # the now-fuller context (predictor-corrector). "low_confidence" targets the
        # shakiest commits; "random" is the stable default.
        if corrector_frac > 0 and (i + 1) % corrector_every == 0:
            committed = x != mask_id
            if corrector_mode == "low_confidence":
                rm = torch.zeros_like(committed)
                for b in range(n):
                    cand = committed[b].nonzero().squeeze(-1)
                    if cand.numel() == 0:
                        continue
                    k = max(1, int(round(corrector_frac * cand.numel())))
                    low = commit_conf[b, cand].topk(k, largest=False).indices
                    rm[b, cand[low]] = True
            else:  # "random"
                rm = (torch.rand(n, seq_len, device=device) < corrector_frac) & committed
            if rm.any():
                x_in = x.masked_fill(rm, mask_id)
                t_c = scheduler.t_from_mask_prob((x_in == mask_id).float().mean(dim=1))
                probs_c = (model(x_in, t_c) / temperature).softmax(-1)
                pred_c = torch.multinomial(
                    probs_c.reshape(-1, probs_c.size(-1)), 1).reshape(n, seq_len)
                # Track the corrected confidence so a fixed position is not re-masked
                # every pass (which would starve it and invite drift/loops).
                commit_conf = torch.where(rm, probs_c.max(-1).values, commit_conf)
                x = torch.where(rm, pred_c, x)

    model.train()
    if return_ids:
        return x
    return [tokenizer.decode(seq, skip_special_tokens=True) for seq in x]


@torch.no_grad()
def refine(model, scheduler, x, passes=8, frac=0.15, temperature=0.9):
    """Regenerate already-committed tokens via gentle random remasking.

    Each pass re-masks a random ``frac`` of positions and re-predicts them
    conditioned on the rest, so the model can fix tokens it committed before the
    full context existed (e.g. an inconsistent character name). Re-masking a
    *small random* fraction is the stable way to do this: re-masking by confidence
    instead collapses into repetitive high-confidence loops ("Mrs Mrs Mrs...").

    ``x`` is an (n, L) id tensor (e.g. from ``sample(..., return_ids=True)``);
    returns the refined ids.
    """
    model.eval()
    mask_id = scheduler.mask_id
    n, L = x.shape
    for _ in range(passes):
        remask = torch.rand(n, L, device=x.device) < frac
        x_in = x.masked_fill(remask, mask_id)
        # In-distribution t for the actual masked fraction (= sigma^{-1}(frac) for
        # the cosine schedule), not frac used directly as t.
        t = scheduler.t_from_mask_prob((x_in == mask_id).float().mean(dim=1))
        probs = (model(x_in, t) / temperature).softmax(-1)
        pred = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1).reshape(n, L)
        x = torch.where(remask, pred, x)
    return x


def _load_tokenizer(source):
    """Load the byte-level BPE tokenizer from a Hub repo id or local directory."""
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(source)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


class DiffusionLM:
    """High-level inference pipeline: model + tokenizer + absorbing scheduler.

    Bundles everything needed to turn the all-``[MASK]`` prior into text, so
    generation is a single call::

        lm = DiffusionLM.from_pretrained("tchauffi/diffusionlm-from-scratch")
        for story in lm.generate(n=4, seq_len=80, temperature=0.9):
            print(story)
    """

    def __init__(self, model, tokenizer, scheduler=None):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.scheduler = scheduler or AbsorbingScheduler(mask_id=tokenizer.mask_token_id)

    @classmethod
    def from_pretrained(cls, source=DEFAULT_REPO, tokenizer=None, device=None):
        """Load model + tokenizer from a Hub repo id or local checkpoint.

        ``source`` is the model: a Hub repo id (e.g.
        ``"tchauffi/diffusionlm-from-scratch"``) or a local ``*.pt`` checkpoint.
        ``tokenizer`` defaults to the same location; pass a repo id / local dir,
        or an already-built tokenizer, to override. ``device`` defaults to CUDA
        when available, else CPU.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = DiT.from_pretrained(source, device=device)
        if tokenizer is None or isinstance(tokenizer, str):
            tokenizer = _load_tokenizer(tokenizer or source)
        return cls(model, tokenizer)

    @torch.no_grad()
    def generate(self, n=4, seq_len=80, steps=None, temperature=0.9,
                 order="confidence", return_ids=False, **kwargs):
        """Generate ``n`` samples from the all-``[MASK]`` prior.

        Returns a list of decoded strings (or the ``(n, seq_len)`` id tensor when
        ``return_ids=True``). ``steps`` defaults to ``seq_len`` (one token per
        step). Extra keyword args pass straight through to :func:`sample`
        (``reveal_schedule``, ``corrector_frac``, ``confidence_threshold``, ...).
        """
        return sample(self.model, self.scheduler, self.tokenizer, seq_len,
                      n=n, steps=steps or seq_len, temperature=temperature,
                      order=order, return_ids=return_ids, **kwargs)

    def refine(self, x, passes=8, frac=0.15, temperature=0.9):
        """Refine an ``(n, L)`` id tensor by gentle random remasking (see :func:`refine`)."""
        return refine(self.model, self.scheduler, x, passes=passes, frac=frac,
                      temperature=temperature)
