"""Capture real generation trajectories from the trained masked-diffusion DiT.

Runs the parallel sampler and records, at every step, an *event stream* that drives
the website animation: each frame is a list of ``[position, token_id]`` ops, where
``token_id == -1`` means the position was re-masked. Plain orders only ever reveal
(monotonic), but the predictor-corrector ("corrected") order also re-masks the
lowest-confidence commits and re-predicts them, so positions can change mid-run.

Usage:
    uv run python scripts/capture_trajectories.py \
        --ckpt runs/dit-full-cont/final.pt \
        --tokenizer tinystories_tokenizer \
        --out docs/trajectories.json \
        --n 96 --keep 6 --seq-len 80 --temperature 0.9
"""

import argparse
import json
import math

import torch

from diffusionlm_from_scratch.model import DiT, DiTConfig
from diffusionlm_from_scratch.scheduler import AbsorbingScheduler
from diffusionlm_from_scratch.dataset import build_tokenizer

MASK_OP = -1   # sentinel id in an event meaning "re-mask this position"


@torch.no_grad()
def capture(model, scheduler, seq_len, n, steps, temperature, order="confidence",
            reveal_schedule="cosine", corrector_frac=0.0, corrector_every=3,
            corrector_mode="low_confidence"):
    """Sampler that records a per-frame event stream.

    Returns ``(x, frames, confs)``:
      - ``x``: (n, seq_len) final ids.
      - ``frames[b]``: ordered list of frames; each frame is a list of ``[p, id]``
        ops (id == -1 -> re-masked). Replaying the frames reproduces ``x[b]``.
      - ``confs[b]``: commit confidences (for readability scoring).

    ``order`` picks which masked positions to commit each step (``confidence`` =
    surest first; ``confidence_weighted`` = confidence-biased but spread;
    ``random`` = scattered). ``corrector_frac > 0`` adds predictor-corrector
    passes that re-mask the lowest-confidence commits and re-predict them.
    """
    model.eval()
    mask_id = scheduler.mask_id
    device = next(model.parameters()).device
    x = torch.full((n, seq_len), mask_id, device=device, dtype=torch.long)
    commit_conf = torch.zeros(n, seq_len, device=device)

    frames = [[] for _ in range(n)]
    confs = [[] for _ in range(n)]

    def emit(per_sample_ops):
        for b in range(n):
            if per_sample_ops[b]:
                frames[b].append(per_sample_ops[b])

    for i in range(steps):
        still = x == mask_id
        if not still.any() and corrector_frac == 0:
            break

        # In-distribution conditioning: t = sigma^{-1}(current masked fraction).
        m = still.float().mean(dim=1)
        t_used = scheduler.t_from_mask_prob(m)
        probs = (model(x, t_used) / temperature).softmax(-1)
        pred = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1).reshape(n, -1)
        conf = probs.max(-1).values

        # PREDICTOR: reveal more tokens (while any remain masked).
        if still.any():
            conf_m = conf.masked_fill(~still, -1.0)
            r = (i + 1) / steps
            if reveal_schedule == "cosine":
                target_unmasked = (1.0 - math.cos(math.pi / 2 * r)) * seq_len
            else:
                target_unmasked = r * seq_len
            step_ops = [[] for _ in range(n)]
            for b in range(n):
                k = max(1, int(round(target_unmasked - int((~still[b]).sum()))))
                k = min(k, int(still[b].sum()))
                if k <= 0:
                    continue
                if order == "confidence":
                    idx = conf_m[b].topk(k).indices
                elif order == "confidence_weighted":
                    idx = torch.multinomial(conf_m[b].clamp(min=0.0), k, replacement=False)
                else:  # "random"
                    cand = still[b].nonzero().squeeze(-1)
                    idx = cand[torch.randperm(cand.numel(), device=device)[:k]]
                x[b, idx] = pred[b, idx]
                commit_conf[b, idx] = conf[b, idx]
                for p in idx.tolist():
                    step_ops[b].append([p, int(pred[b, p])])
                    confs[b].append(round(float(conf[b, p]), 3))
            emit(step_ops)

        # CORRECTOR: re-mask the lowest-confidence commits, re-predict with fuller
        # context. Emitted as two frames (re-mask, then re-fill) so the animation
        # shows tokens flicker back to [MASK] and resolve to (often) new words.
        if corrector_frac > 0 and (i + 1) % corrector_every == 0:
            committed = x != mask_id
            rm = torch.zeros_like(committed)
            for b in range(n):
                cand = committed[b].nonzero().squeeze(-1)
                if cand.numel() == 0:
                    continue
                k = max(1, int(round(corrector_frac * cand.numel())))
                low = commit_conf[b, cand].topk(k, largest=False).indices
                rm[b, cand[low]] = True
            if rm.any():
                remask_ops = [[] for _ in range(n)]
                for b in range(n):
                    for p in rm[b].nonzero().squeeze(-1).tolist():
                        remask_ops[b].append([p, MASK_OP])
                x_in = x.masked_fill(rm, mask_id)
                t_c = scheduler.t_from_mask_prob((x_in == mask_id).float().mean(dim=1))
                probs_c = (model(x_in, t_c) / temperature).softmax(-1)
                pred_c = torch.multinomial(
                    probs_c.reshape(-1, probs_c.size(-1)), 1).reshape(n, seq_len)
                commit_conf = torch.where(rm, probs_c.max(-1).values, commit_conf)
                x = torch.where(rm, pred_c, x)
                refill_ops = [[] for _ in range(n)]
                for b in range(n):
                    for p in rm[b].nonzero().squeeze(-1).tolist():
                        refill_ops[b].append([p, int(x[b, p])])
                emit(remask_ops)
                emit(refill_ops)

    return x, frames, confs


def clean_token_text(tok_str):
    """Byte-level BPE token -> (display text, leading_space).

    'Ġ' marks a leading space and 'Ċ' a newline; the space is returned as a flag
    (not baked into the text) so the renderer controls spacing.
    """
    leading_space = tok_str.startswith("Ġ")
    body = tok_str[1:] if leading_space else tok_str
    text = body.replace("Ġ", " ").replace("Ċ", "\n")
    return text, leading_space


def readability_score(text, mean_conf):
    """Heuristic legibility score for surfacing the cleanest honest samples.

    Hard-rejects degenerate loops (character runs, words repeated 3x); otherwise
    rewards a capitalized opening and penalizes adjacent dup words, low vocab
    diversity, repeated bigrams, and fragment spam.
    """
    import re
    words = text.split()
    if len(words) < 6:
        return -10.0
    lower = [w.lower().strip('.,!?"\'') for w in words]
    if re.search(r'(.)\1{4,}', text):
        return -10.0
    if any(lower[i] and lower[i] == lower[i + 1] == lower[i + 2]
           for i in range(len(lower) - 2)):
        return -10.0
    score = min(mean_conf, 0.72)
    if text[:1].isupper():
        score += 0.08
    dup_adj = sum(1 for a, b in zip(lower, lower[1:]) if a and a == b)
    score -= 0.6 * dup_adj / len(words)
    score -= 0.4 * (1.0 - len(set(lower)) / len(lower))
    bigrams = list(zip(lower, lower[1:]))
    if bigrams:
        score -= 0.6 * (1.0 - len(set(bigrams)) / len(bigrams))
    nl = text.count("\n")
    score -= 0.04 * max(0, nl - len(words) / 12)
    score += 0.03 * min(len(words), 60) / 60
    return score


def select_samples(x, frames, confs, tok, eos_id, keep):
    """Score, trim, and serialize the best samples as event streams.

    Each kept sample is ``{"len": cut, "events": [[ [p,id], ... ], ... ], "conf": ..}``
    with events restricted to positions < cut (the first-EOS trim).
    """
    n = x.size(0)
    scored = []
    for b in range(n):
        ids = x[b].tolist()
        cut = ids.index(eos_id) if eos_id in ids else len(ids)
        cut = max(cut, 8)
        mean_conf = sum(confs[b]) / max(len(confs[b]), 1)
        text = tok.decode(ids[:cut], skip_special_tokens=True).strip()
        scored.append((readability_score(text, mean_conf), mean_conf, b, cut))
    scored.sort(reverse=True)

    samples = []
    for _, mean_conf, b, cut in scored[:keep]:
        events = []
        for frame in frames[b]:
            ops = [[p, idv] for p, idv in frame if p < cut]
            if ops:
                events.append(ops)
        samples.append({"len": cut, "events": events, "conf": round(mean_conf, 3)})
        preview = tok.decode(x[b].tolist()[:cut], skip_special_tokens=True).strip().replace("\n", " / ")
        n_corr = sum(1 for fr in events for p, idv in fr if idv == MASK_OP)
        print(f"  kept b={b} conf={mean_conf:.3f} corrections={n_corr}: {preview[:100]}")
    return samples


def build_vocab(out_orders, tok):
    """Collect every token id referenced by any event -> compact {id: [text, s]}."""
    ids = set()
    for samples in out_orders.values():
        for s in samples:
            for frame in s["events"]:
                for p, idv in frame:
                    if idv >= 0:
                        ids.add(idv)
    vocab = {}
    for idv in ids:
        text, lead = clean_token_text(tok.convert_ids_to_tokens([idv])[0])
        vocab[str(idv)] = [text, 1 if lead else 0]
    return vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/dit-full-cont/final.pt")
    ap.add_argument("--tokenizer", default="tinystories_tokenizer")
    ap.add_argument("--out", default="docs/trajectories.json")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--keep", type=int, default=6)
    ap.add_argument("--seq-len", type=int, default=80)
    ap.add_argument("--steps", type=int, default=None, help="sampler steps; default = seq_len")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--corrector-frac", type=float, default=0.12)
    ap.add_argument("--corrector-every", type=int, default=3)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, mask_id, vocab_size = build_tokenizer(args.tokenizer)
    eos_id = tok.eos_token_id

    ck = torch.load(args.ckpt, map_location="cpu")
    model = DiT(DiTConfig(**ck["config"])).to(device)
    model.load_state_dict(ck.get("model", ck["raw"]))
    print(f"loaded {args.ckpt}: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params, vocab {vocab_size}")

    steps = args.steps or args.seq_len
    scheduler = AbsorbingScheduler(mask_id=mask_id)

    # Three plain reveal orders + a predictor-corrector ("corrected") run.
    specs = [
        ("confidence", dict(order="confidence")),
        ("confidence_weighted", dict(order="confidence_weighted")),
        ("random", dict(order="random")),
        ("corrected", dict(order="confidence", corrector_frac=args.corrector_frac,
                           corrector_every=args.corrector_every,
                           corrector_mode="low_confidence")),
    ]
    out_orders = {}
    for name, kw in specs:
        torch.manual_seed(args.seed)              # same start across orders
        print(f"order={name}")
        x, frames, confs = capture(model, scheduler, seq_len=args.seq_len, n=args.n,
                                   steps=steps, temperature=args.temperature, **kw)
        out_orders[name] = select_samples(x, frames, confs, tok, eos_id, args.keep)

    out = {
        "model": "dit-full (142M, h768/d12, vocab 8192, TinyStories)",
        "seq_len": args.seq_len,
        "steps": steps,
        "temperature": args.temperature,
        "corrector": {"frac": args.corrector_frac, "every": args.corrector_every},
        "vocab": build_vocab(out_orders, tok),
        "orders": out_orders,
    }
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {sum(len(v) for v in out_orders.values())} trajectories "
          f"({', '.join(k for k, _ in specs)}) -> {args.out}")


if __name__ == "__main__":
    main()
