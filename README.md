# diffusionlm-from-scratch

A masked (absorbing-state) **diffusion language model**, built and trained from
scratch on TinyStories. Instead of writing left-to-right one token at a time, it
starts from a sequence of pure `[MASK]` and **denoises the whole sequence in
parallel** — committing the tokens it is most confident about first, in whatever
order the meaning falls into place.

- **Course / write-up:** [`RESEARCH.md`](RESEARCH.md) — a from-scratch course on
  discrete/text diffusion (D3PM → absorbing-state → sampling).
- **Model:** a 142M-parameter DiT (h768 · 12 layers, bidirectional attention,
  adaLN-Zero), 8,192-token byte-level BPE, eval cross-entropy **2.18**.
- **The key finding:** uniform loss weighting (`w(t)=1`), *not* the textbook ELBO
  weight `1/σ(t)`, was what turned word-salad into coherent stories.

## Generate text

The trained model is on the
[Hugging Face Hub](https://huggingface.co/tchauffi/diffusionlm-from-scratch).
`DiffusionLM` bundles the model, tokenizer, and absorbing-state scheduler, so
sampling stories is two lines:

```python
from diffusionlm_from_scratch import DiffusionLM

lm = DiffusionLM.from_pretrained("tchauffi/diffusionlm-from-scratch")
for story in lm.generate(n=4, seq_len=80, temperature=0.9):
    print(story)
```

`from_pretrained` also accepts a local `*.pt` checkpoint, and `generate` exposes
the sampler knobs (`order`, `steps`, `corrector_frac`, `confidence_threshold`, …).

## Demo site

`docs/` is a self-contained static site that **animates real generations** from
the trained model: each token lights up in the exact order the sampler committed
it, tinted by its token ID like tiktoken.

```bash
# serve locally (fetch() needs http, not file://)
cd docs && python -m http.server 8765
# then open http://localhost:8765
```

Deploy by pointing **GitHub Pages** at the `docs/` folder
(Settings → Pages → Branch: `main` / `docs`).

### Regenerating the animations

Every trajectory in `docs/trajectories.json` is captured from a checkpoint by
replaying the confidence sampler and recording which positions get committed at
each step. By default it pulls the model and tokenizer straight from the
[Hugging Face Hub](https://huggingface.co/tchauffi/diffusionlm-from-scratch):

```bash
uv run python scripts/capture_trajectories.py \
    --out docs/trajectories.json \
    --n 64 --keep 8 --seq-len 84 --temperature 0.85
```

Pass `--ckpt` / `--tokenizer` (a Hub repo id or a local path) to use your own
checkpoint instead.

## Project layout

| Path | What |
|------|------|
| `src/diffusionlm_from_scratch/model.py` | the DiT (timestep embedding, adaLN-Zero blocks) + `from_pretrained` |
| `src/diffusionlm_from_scratch/scheduler.py` | absorbing-state forward corruption + loss weights |
| `src/diffusionlm_from_scratch/inference.py` | `DiffusionLM` pipeline + confidence/predictor-corrector sampler |
| `src/diffusionlm_from_scratch/trainer.py` | training loop, optimizers, EMA, periodic eval/sampling |
| `src/diffusionlm_from_scratch/dataset.py` | TinyStories tokenization pipeline |
| `scripts/capture_trajectories.py` | exports denoising trajectories for the site |
| `docs/` | the animated showcase site |

> Samples come from a small model on a children's-story corpus — expect simple
> vocabulary and the occasional slip. The remarkable part is that grammatical,
> structured stories emerge at all from a process that fills positions out of order.
