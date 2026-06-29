"""Training and sampling for the masked (absorbing-state) text-diffusion DiT.

Mixed precision, device placement, and TensorBoard logging are handled by
`accelerate`. The objective is the schedule-weighted masked cross-entropy from
RESEARCH.md (sections 5-6); generation uses the confidence-based parallel
sampler from section 6.
"""

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from accelerate import Accelerator

from .model import DiT, DiTConfig
from .scheduler import AbsorbingScheduler
from .dataset import DataConfig, get_dataloader


@dataclass
class TrainConfig:
    epochs: int = 6
    batch_size: int = 64
    optimizer: str = "adamw"             # "adamw" | "muon" (Muon body + AdamW aux)
    lr: float = 3e-4
    warmup_steps: int = 500              # linear warmup, then cosine decay
    min_lr_ratio: float = 0.1            # cosine floor as a fraction of lr
    weight_decay: float = 0.01
    loss_weight_clamp: float = 10.0      # cap on the schedule weight 1/sigma(t)
    # "uniform" trains the high-mask regime generation depends on; it was the
    # breakthrough over the ELBO "inv_sigma" weight (see RESEARCH / project notes).
    weight_mode: str = "uniform"         # "uniform" | "inv_sigma" | "sigma"
    # EMA note: standard for diffusion, but on this masked-text model it did NOT
    # help -- the smoothed weights are more peaked, so at low temperature the
    # confidence sampler loops more than the raw weights. Kept (checkpoints store
    # both EMA under "model" and raw under "raw"); sampling from "raw" is fine too.
    use_ema: bool = True                 # sample/save from an EMA of the weights
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    mixed_precision: str = "bf16"        # "no" | "fp16" | "bf16"
    # DataLoader workers. >0 speeds loading but can grow host RAM over a long epoch
    # on a large dataset (worker copy-on-read) -> OOM. Use 0 for big runs.
    num_workers: int = 2
    log_every: int = 50
    eval_every: int = 1000               # steps; 0 disables periodic eval
    eval_batches: int = 50               # held-out batches per eval
    sample_every: int = 1000             # steps; 0 disables periodic sampling
    sample_n: int = 4                    # number of samples to generate/log
    sample_steps: int = None             # sampler steps; None -> seq_len (1 token/step)
    # T~0.9 reads best: lower (0.7) makes the confidence sampler loop on
    # high-confidence tokens ("splash and splash and splash"); higher tends to drift.
    sample_temperature: float = 0.9
    save_every: int = 2000               # steps; 0 disables periodic checkpoints
    output_dir: str = "runs/dit-text"
    seed: int = 0


def compute_loss(model, scheduler: AbsorbingScheduler, x, attention_mask=None):
    """Schedule-weighted masked cross-entropy = E[ w(t) * CE on masked tokens ].

    Returns ``(loss, masked_token_count)``; the count lets the caller aggregate
    a proper per-token loss for a perplexity bound.
    """
    B = x.size(0)
    t = scheduler.sample_t(B, x.device)
    x_t, mask = scheduler.corrupt(x, t, attention_mask=attention_mask)

    logits = model(x_t, t)                                       # (B, L, V)
    ce = F.cross_entropy(logits.transpose(1, 2), x, reduction="none")  # (B, L)

    denom = mask.sum(1).clamp(min=1)
    per_ex = (ce * mask).sum(1) / denom                         # mean CE over masked
    loss = (per_ex * scheduler.loss_weight(t)).mean()
    return loss, mask.sum()


def build_optimizers(model, cfg: TrainConfig):
    """Return a list of optimizers for the training loop.

    For ``"muon"``: the 2D transformer-body matrices (attention, MLP, adaLN
    projections) are orthogonalized by Muon, while everything else -- token and
    positional embeddings, the timestep MLP, the output head, and all biases --
    is handled by AdamW (Muon only supports 2D params and works best when
    embeddings/head stay on Adam). ``adjust_lr_fn="match_rms_adamw"`` lets Muon
    reuse the AdamW-tuned lr/weight decay.
    """
    if cfg.optimizer == "adamw":
        return [torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)]
    if cfg.optimizer == "muon":
        muon_params, adamw_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("blocks.") and p.ndim == 2:
                muon_params.append(p)
            else:
                adamw_params.append(p)
        muon = torch.optim.Muon(muon_params, lr=cfg.lr,
                                weight_decay=cfg.weight_decay,
                                adjust_lr_fn="match_rms_adamw")
        adamw = torch.optim.AdamW(adamw_params, lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
        return [muon, adamw]
    raise ValueError(f"unknown optimizer: {cfg.optimizer!r}")


@torch.no_grad()
def evaluate(model, scheduler, dataloader, max_batches=50):
    """Held-out token-level cross-entropy and its perplexity bound."""
    model.eval()
    total_ce, total_tok = 0.0, 0
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        x = batch["input_ids"]
        attn = batch.get("attention_mask")
        t = scheduler.sample_t(x.size(0), x.device)
        x_t, mask = scheduler.corrupt(x, t, attention_mask=attn)
        logits = model(x_t, t)
        ce = F.cross_entropy(logits.transpose(1, 2), x, reduction="none")
        total_ce += (ce * mask).sum().item()
        total_tok += mask.sum().item()
    model.train()
    mean_ce = total_ce / max(total_tok, 1)
    return mean_ce, float(torch.exp(torch.tensor(mean_ce)))


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


class EMA:
    """Exponential moving average of model weights, used for sampling.

    Diffusion models sample from a smoothed copy of the weights rather than the
    raw training weights; this is standard for DiT and usually gives cleaner,
    more stable samples. Holds a shadow copy and supports temporarily swapping it
    into the live model (``store``/``copy_to``/``restore``) for sampling/saving.
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self._backup = None

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        """Load EMA weights into ``model`` (after stashing the raw ones)."""
        self._backup = {n: p.detach().clone()
                        for n, p in model.named_parameters() if p.requires_grad}
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model):
        """Undo :meth:`copy_to`, restoring the raw training weights."""
        if self._backup is None:
            return
        for n, p in model.named_parameters():
            if n in self._backup:
                p.data.copy_(self._backup[n])
        self._backup = None


def save_checkpoint(accelerator, model, model_cfg, tag, output_dir, ema=None):
    """Save unwrapped model weights + config to ``output_dir/tag.pt`` (main only).

    When ``ema`` is given, its smoothed weights are saved under ``"model"`` (the
    weights you actually want to sample from) and the raw weights under ``"raw"``.
    """
    if not accelerator.is_main_process:
        return
    import os
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{tag}.pt")
    raw = accelerator.unwrap_model(model).state_dict()
    payload = {"model": raw, "config": vars(model_cfg)}
    if ema is not None:
        payload = {"model": ema.shadow, "raw": raw, "config": vars(model_cfg)}
    accelerator.save(payload, path)
    accelerator.print(f"  saved checkpoint: {path}")


def train(train_cfg: TrainConfig = TrainConfig(),
          data_cfg: DataConfig = DataConfig(),
          model_cfg: DiTConfig = None,
          schedule: str = "cosine",
          resume_from: str = None):
    """End-to-end training loop with accelerate + TensorBoard.

    ``resume_from``: path to a checkpoint to initialize the model weights from
    (its ``"raw"`` training weights, falling back to ``"model"``). Optimizer/EMA
    state is *not* restored -- this is a weight-init resume, so pair it with a
    fresh (shorter) LR schedule continuing from where the original left off.
    """
    accelerator = Accelerator(
        mixed_precision=train_cfg.mixed_precision,
        log_with="tensorboard",
        project_dir=train_cfg.output_dir,
    )
    accelerator.init_trackers("dit-text", config=vars(train_cfg))
    if train_cfg.seed is not None:
        from accelerate.utils import set_seed
        set_seed(train_cfg.seed)

    dataloader, tokenizer, mask_id, vocab_size = get_dataloader(
        data_cfg, batch_size=train_cfg.batch_size, shuffle=True,
        num_workers=train_cfg.num_workers,
    )

    # Held-out split for the perplexity-bound curve. Reuse the same tokenizer.
    val_loader = None
    if train_cfg.eval_every:
        from dataclasses import replace
        val_cfg = replace(data_cfg, split=data_cfg.val_split)
        val_loader, _, _, _ = get_dataloader(
            val_cfg, batch_size=train_cfg.batch_size, shuffle=False,
            tokenizer=tokenizer, num_workers=0,
        )

    if model_cfg is None:
        model_cfg = DiTConfig(vocab_size=vocab_size, max_seq_len=data_cfg.seq_len)
    model = DiT(model_cfg)
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location="cpu")
        model.load_state_dict(ckpt.get("raw", ckpt["model"]))  # raw weights to continue
        accelerator.print(f"resumed weights from {resume_from}")
    scheduler = AbsorbingScheduler(mask_id=mask_id, schedule=schedule,
                                   max_weight=train_cfg.loss_weight_clamp,
                                   weight_mode=train_cfg.weight_mode)

    optimizers = build_optimizers(model, train_cfg)

    # Linear warmup -> cosine decay to min_lr_ratio * lr, as an LR *multiplier*.
    total_steps = train_cfg.epochs * len(dataloader)

    def lr_multiplier(step):
        if step < train_cfg.warmup_steps:
            return step / max(1, train_cfg.warmup_steps)
        progress = (step - train_cfg.warmup_steps) / max(1, total_steps - train_cfg.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return train_cfg.min_lr_ratio + (1.0 - train_cfg.min_lr_ratio) * cosine

    lr_schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, lr_multiplier)
                     for opt in optimizers]

    prepared = accelerator.prepare(model, dataloader, *optimizers, *lr_schedulers)
    model, dataloader = prepared[0], prepared[1]
    n_opt = len(optimizers)
    optimizers = list(prepared[2:2 + n_opt])
    lr_schedulers = list(prepared[2 + n_opt:])
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)

    unwrapped = accelerator.unwrap_model(model)
    ema = EMA(unwrapped, decay=train_cfg.ema_decay) if train_cfg.use_ema else None

    n_params = sum(p.numel() for p in model.parameters())
    accelerator.print(f"params: {n_params/1e6:.1f}M | vocab: {vocab_size} | "
                      f"schedule: {schedule} | weight_mode: {train_cfg.weight_mode} | "
                      f"ema: {train_cfg.ema_decay if ema else 'off'} | "
                      f"precision: {train_cfg.mixed_precision}")

    global_step = 0
    for epoch in range(train_cfg.epochs):
        model.train()
        for batch in dataloader:
            x = batch["input_ids"]
            attn = batch.get("attention_mask")
            with accelerator.accumulate(model):
                loss, _ = compute_loss(model, scheduler, x, attention_mask=attn)
                accelerator.backward(loss)
                if train_cfg.grad_clip:
                    accelerator.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                for opt in optimizers:
                    opt.step()
                for sched in lr_schedulers:
                    sched.step()
                for opt in optimizers:
                    opt.zero_grad()
                if ema is not None and accelerator.sync_gradients:
                    ema.update(unwrapped)

            if global_step % train_cfg.log_every == 0:
                cur_lr = optimizers[0].param_groups[0]["lr"]
                accelerator.log({"train/loss": loss.item(), "train/lr": cur_lr},
                                step=global_step)
                accelerator.print(f"ep {epoch} step {global_step} "
                                  f"loss {loss.item():.4f} lr {cur_lr:.2e}")

            if val_loader is not None and global_step % train_cfg.eval_every == 0:
                ce, ppl = evaluate(model, scheduler, val_loader,
                                   max_batches=train_cfg.eval_batches)
                accelerator.log({"eval/ce": ce, "eval/ppl": ppl}, step=global_step)
                accelerator.print(f"  [eval] step {global_step} ce {ce:.4f} ppl {ppl:.2f}")

            if train_cfg.sample_every \
                    and global_step % train_cfg.sample_every == 0 \
                    and accelerator.is_main_process:
                if ema is not None:
                    ema.copy_to(unwrapped)        # sample from the smoothed weights
                samples = sample(unwrapped, scheduler, tokenizer,
                                 seq_len=model_cfg.max_seq_len,
                                 n=train_cfg.sample_n,
                                 steps=train_cfg.sample_steps or model_cfg.max_seq_len,
                                 temperature=train_cfg.sample_temperature)
                if ema is not None:
                    ema.restore(unwrapped)
                for j, s in enumerate(samples):
                    accelerator.print(f"  sample[{j}]: {s[:160]}")
                # Log to TensorBoard as text so generation can be scrubbed over steps.
                writer = accelerator.get_tracker("tensorboard", unwrap=True)
                if writer is not None:
                    md = "\n\n".join(f"**[{j}]** {s}" for j, s in enumerate(samples))
                    writer.add_text("samples", md, global_step)

            if train_cfg.save_every and global_step > 0 \
                    and global_step % train_cfg.save_every == 0:
                save_checkpoint(accelerator, model, model_cfg,
                                f"step_{global_step}", train_cfg.output_dir, ema=ema)

            global_step += 1

    save_checkpoint(accelerator, model, model_cfg, "final", train_cfg.output_dir, ema=ema)
    accelerator.end_training()
    accelerator.print("training complete")
    if ema is not None:                      # return the EMA weights for sampling
        ema.copy_to(unwrapped)
    return unwrapped, tokenizer, scheduler


if __name__ == "__main__":
    train()
