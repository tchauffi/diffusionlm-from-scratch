# Diffusion Language Models: A Course from Scratch

A self-contained course for someone who already understands continuous (image)
diffusion and wants to understand, build, and reason about diffusion models for
text. It builds toward a working demonstrator trained on TinyStories using the
absorbing-state (masked) formulation, while situating that choice in the broader
research landscape.

**Prerequisites:** comfort with continuous diffusion (forward noising, the
reverse denoiser, epsilon-prediction, DDPM/DDIM sampling), basic probability,
PyTorch, and the transformer architecture.

> **Rendering note.** Equations use LaTeX in `$...$` (inline) and `$$...$$`
> (display) delimiters. These render on GitHub, in VS Code's markdown preview,
> Obsidian, Jupyter, and most MathJax/KaTeX-enabled viewers. A plain text editor
> will show the raw LaTeX source.

---

## Table of contents

1. Why text breaks continuous diffusion
2. The discrete diffusion framework (D3PM)
3. The two corruption processes: uniform and absorbing
4. Noise schedules for discrete data
5. The training objective: from ELBO to masked cross-entropy
6. The reverse process and sampling
7. Score-based discrete diffusion (SEDD)
8. Continuous-embedding approaches (the other family)
9. Building the demonstrator on TinyStories
10. Evaluation
11. Practical failure modes and fixes
12. Where the field is going
13. Annotated reading list
14. Exercises

---

## 1. Why text breaks continuous diffusion

In image diffusion you exploit a property text does not have: pixels live in a
continuous space where small perturbations are meaningful. Adding a little
Gaussian noise to a pixel gives a slightly different but still valid pixel. The
whole machinery — the reparameterization trick, epsilon-prediction, the
closed-form Gaussian posterior — depends on this.

Text is categorical. A token is an index into a vocabulary; there is no
meaningful "halfway between token 4,812 and token 9,001." You cannot add a
fractional amount of Gaussian noise to the index "cat" and get something
sensible. This kills three things at once:

- There is no reparameterization trick over a categorical variable, so you can't
  push gradients through sampling the same way.
- There is no additive noise vector to predict, so epsilon-prediction has no
  direct analogue.
- The forward and reverse transitions are distributions over a discrete set, so
  the L2 losses of image diffusion become KL divergences between categorical
  distributions.

There are two ways out, and they define the two families of text diffusion:

1. **Work in the discrete space directly.** Define corruption as a stochastic
   jump from one token to another (or to a special state). This is *discrete
   diffusion* and is the main focus of this course.
2. **Map tokens into a continuous space first**, run ordinary Gaussian diffusion
   on the embeddings, then map back. This is the *continuous-embedding* family
   (Diffusion-LM and successors), covered in section 8.

The discrete route has become dominant for strong results, so we develop it
first and in the most depth.

---

## 2. The discrete diffusion framework (D3PM)

The foundational reference is **D3PM** (Austin et al., 2021, "Structured
Denoising Diffusion Models in Discrete State-Spaces"). It generalizes the DDPM
recipe to categorical data.

### Setup

Let the vocabulary have $K$ categories. Represent a single token at timestep $t$
as a one-hot row vector $x_t$ of length $K$. The forward process is a Markov
chain defined by **transition matrices** $Q_t$, each $K \times K$, where entry
$[Q_t]_{ij}$ is the probability that a token in state $i$ moves to state $j$ at
step $t$:

$$q(x_t \mid x_{t-1}) = \mathrm{Cat}\!\left(x_t \,;\, p = x_{t-1} Q_t\right)$$

Because the chain is Markov and the steps compose by matrix multiplication, the
$t$-step marginal is also closed form:

$$q(x_t \mid x_0) = \mathrm{Cat}\!\left(x_t \,;\, p = x_0 \bar{Q}_t\right),
\qquad \bar{Q}_t = Q_1 Q_2 \cdots Q_t$$

This $\bar{Q}_t$ is the discrete analogue of the cumulative $\bar{\alpha}_t$ schedule you know
from DDPM: it tells you the distribution of the corrupted token given the
original, for any $t$, without simulating intermediate steps. That is what makes
training efficient — you can jump straight to a random timestep.

### The posterior

The reverse process needs the posterior $q(x_{t-1} \mid x_t, x_0)$. By Bayes' rule,
and because everything is categorical, this is available in closed form:

$$q(x_{t-1} \mid x_t, x_0) \;\propto\; \left(x_t Q_t^{\top}\right) \odot \left(x_0 \bar{Q}_{t-1}\right)$$

($\odot$ is elementwise product, then normalize.) The model is trained so that its
parameterized reverse $p_\theta(x_{t-1} \mid x_t)$ matches this posterior. The key
modeling choice in D3PM — and the one that carries all the way to your
demonstrator — is that the network predicts $x_0$ (a distribution over the
clean token) rather than predicting the transition directly. You then plug that
predicted $x_0$ into the posterior formula above.

This is the discrete echo of x0-prediction in continuous diffusion: instead of
regressing the noise, the network classifies the original token.

### Why the transition matrix matters

Everything interesting about a discrete diffusion model is encoded in the choice
of $Q_t$. Different $Q_t$ give qualitatively different corruption processes,
the way different $\beta_t$ schedules and noise types would in the continuous case.
The next section covers the two that matter.

---

## 3. The two corruption processes: uniform and absorbing

### Uniform (multinomial) corruption

Here a token, with some probability per step, is resampled uniformly at random
from the whole vocabulary. The transition matrix is

$$Q_t = (1 - \beta_t)\, I + \beta_t\, \tfrac{1}{K}\, \mathbf{1}\mathbf{1}^{\top}$$

so with probability $(1 - \beta_t)$ the token stays, and with probability $\beta_t$ it
jumps to a uniformly random token. As $t \to T$, every token has been resampled
many times and the marginal $q(x_T \mid x_0)$ converges to the uniform
distribution over the vocabulary — the discrete analogue of "pure noise."

This is exactly **Multinomial Diffusion** (Hoogeboom et al., 2021), which
appeared in parallel with D3PM and is the uniform special case. It is the
cleanest thing to reason about, but for text it tends to underperform, because
the intermediate states are sequences full of random real tokens — locally
plausible garbage that is hard to denoise.

### Absorbing-state corruption (masking)

Here a token, with some probability per step, is replaced by a special absorbing
`[MASK]` state and then never changes again. Adding `[MASK]` as category $m$, the
transition matrix keeps each token fixed with probability $(1 - \beta_t)$ and sends
it to `[MASK]` with probability $\beta_t$:

$$[Q_t]_{ii} = 1 - \beta_t \quad (\text{stay}), \qquad
[Q_t]_{im} = \beta_t \quad (\text{absorb}), \qquad
[Q_t]_{mm} = 1 \quad (\text{absorbing})$$

As $t \to T$, every token is absorbed and $q(x_T \mid x_0)$ is the all-`[MASK]`
sequence — a single deterministic state, which is a very convenient prior to
sample from. This is **absorbing-state D3PM**, and it is the formulation your
demonstrator uses.

Two properties make it the practical favorite for text:

- **The posterior is trivial for unmasked positions.** If a token is not
  `[MASK]` at time $t$, then in the forward process it was never corrupted, so
  its $x_0$ is known with certainty. All the modeling effort goes into the
  masked positions only.
- **It generalizes BERT.** A single absorbing step is exactly masked language
  modeling. Discrete diffusion turns the fixed BERT masking ratio into a
  continuum of ratios indexed by $t$, and adds a principled iterative sampler on
  top. This is why the formulation feels familiar and trains stably.

This connection was later sharpened into clean, easy-to-implement objectives by
**MDLM** (Sahoo et al., 2024, "Simple and Effective Masked Diffusion Language
Models") and the concurrent work of **Shi et al., 2024**. Those papers are the
most practical entry point for a from-scratch build because the loss collapses
to a weighted masked cross-entropy.

### Which to use

For a demonstrator: absorbing/masking. It is simpler, trains more stably, gives
better samples, and connects to intuition you already have. Use uniform
corruption only if you specifically want to study the more general case.

---

## 4. Noise schedules for discrete data

The schedule controls how fast corruption accumulates. In the absorbing case the
relevant quantity is the **cumulative masking probability**: the chance that a
given token has been absorbed by time $t$. Call it $\sigma(t)$, with $\sigma(0) = 0$ and
$\sigma(1) = 1$. This is the discrete counterpart of your $\bar{\alpha}_t$ curve, but it
parameterizes a Bernoulli masking rate rather than a Gaussian variance.

Two common choices:

- **Linear:** $\sigma(t) = t$. Corruption accumulates evenly across timesteps;
  training time is spread uniformly over all masking levels.
- **Cosine:** $\sigma(t) = 1 - \cos\!\left(\tfrac{\pi t}{2}\right)$. Masking stays low for small $t$, so the
  model spends more capacity on lightly corrupted sequences (fine-grained
  reconstruction) and compresses the hard, near-fully-masked regime into a
  narrow band near $t = 1$.

For text, cosine-style schedules usually sample slightly better, so it is worth
implementing both and comparing. The continuous-time view (used by MDLM and
SEDD) treats $t$ as a real number in $[0, 1]$ and $\sigma(t)$ as a smooth function,
which removes the need to fix a discrete number of steps at training time — you
sample $t$ uniformly from $[0,1]$ each minibatch. That is the convention the
demonstrator uses.

A subtlety worth internalizing: in continuous diffusion the schedule controls a
signal-to-noise ratio. Here there is no signal-to-noise ratio. A token is either
its original self or `[MASK]`; the schedule only sets the probability of that
discrete flip. The structural role is the same, the mechanism is different.

---

## 5. The training objective: from ELBO to masked cross-entropy

### The general ELBO

Like DDPM, discrete diffusion optimizes a variational bound that decomposes into
per-step KL divergences between categorical distributions:

$$\mathcal{L} = \mathbb{E}_q\!\left[\sum_t \mathrm{KL}\!\left(q(x_{t-1} \mid x_t, x_0)\;\|\;p_\theta(x_{t-1} \mid x_t)\right)\right] + \text{const}$$

In the general D3PM case you compute these KLs using the posterior formula from
section 2 and the network's predicted $x_0$. D3PM also adds an auxiliary
cross-entropy term encouraging the network to predict $x_0$ directly, which
stabilizes training.

### The masked-diffusion simplification

For the absorbing case, the bound collapses into something strikingly simple.
Because unmasked positions have a trivial (certain) posterior, only masked
positions contribute, and the per-step KL reduces to a cross-entropy on the
original token weighted by the schedule. In continuous time the training loss
becomes, up to a constant:

$$\mathcal{L} = \mathbb{E}_{t,\,x_0,\,x_t}\!\left[\frac{w(t)}{|M_t|}\sum_{i \in M_t} -\log p_\theta\!\left(x_0^{\,i} \mid x_t\right)\right]$$

where $M_t$ is the set of masked positions in the corrupted sample, and $w(t)$
is a schedule-dependent weight (for MDLM-style training, $w(t) \propto \sigma'(t)/\sigma(t)$;
a practical and common simplification is $w(t) = 1/\sigma(t)$). In words:

> Mask some tokens according to the schedule at a random $t$. Have a bidirectional
> transformer predict the originals at the masked positions. Take the
> cross-entropy there, reweight by the schedule, average.

That is the entire objective. It is BERT's masked-LM loss with two additions: a
*continuum* of masking ratios indexed by $t$, and a *principled per-$t$
weighting* that makes the whole thing a valid likelihood bound rather than an
ad-hoc auxiliary task. The weighting is what separates masked diffusion from
plain BERT; do not drop it.

### Connecting to the code

In the demonstrator this is literally:

```python
ce = F.cross_entropy(logits.transpose(1, 2), x, reduction="none")  # (B, L)
w  = 1.0 / mask_prob(t).clamp(min=1e-3)                            # schedule weight
per_ex = (ce * mask).sum(1) / mask.sum(1).clamp(min=1)            # masked positions only
loss   = (per_ex * w).mean()
```

`logits` is $p_\theta(x_0)$, `x` is the clean tokens, `mask` selects the masked
positions, and `w` is the schedule reweighting. There is no epsilon anywhere —
the target is the clean token, scored by classification.

---

## 6. The reverse process and sampling

Generation runs the corruption backward. Start from the prior — the all-`[MASK]`
sequence at $t = 1$ — and walk $t$ down to $0$. At each step:

1. Feed the current partially-masked sequence and $t$ to the network.
2. Get $p_\theta(x_0)$ at every masked position.
3. **Commit** some subset of masked positions to actual tokens (sampled from or
   argmaxed over $p_\theta$), leaving the rest as `[MASK]`.
4. Decrease $t$ and repeat.

A committed token stays fixed for the rest of sampling — the reverse process is
the forward absorbing process run backward, so once unmasked it does not return
to `[MASK]` (in the strict version).

Three things to understand:

**Order is not left-to-right.** Every prediction is conditioned on the full
bidirectional context, so the model can fill position 6 before position 1. This
is the source of parallelism: all masked positions are scored at once, and you
can commit many per step. This is the fundamental difference from autoregressive
generation.

**Steps are a sampling-time knob, not a trained constant.** How many positions
you commit per step (the `frac` term in the code) controls a speed/quality
tradeoff, exactly like the number of DDIM steps in image diffusion. More steps,
fewer commits each, better samples.

**Confidence-based unmasking is the standard refinement.** Instead of committing
random masked positions, commit the ones where $p_\theta$ is most peaked (highest max
probability, or lowest entropy). The model locks in what it is surest about first
and leaves ambiguous positions for later steps when more context exists. This
noticeably improves coherence and is what strong masked-diffusion samplers do.

**Remasking** is an optional relaxation (used by SEDD and some others) where a
committed token can be sent back to `[MASK]` if the model becomes less confident
given newly revealed context. The strict absorbing version does not do this;
start strict, add remasking only if you want to study it.

A minimal sampler:

```python
@torch.no_grad()
def sample(model, n=4, steps=64, device="cuda"):
    x = torch.full((n, SEQ_LEN), MASK_ID, device=device)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for i in range(steps):
        t = ts[i].expand(n)
        logits = model(x, t)
        probs = logits.softmax(-1)
        conf, pred = probs.max(-1)                 # confidence per position
        still = x == MASK_ID
        # number of positions to reveal this step
        k = max(1, int(still.sum(1).float().mean() * (ts[i] - ts[i+1]) / ts[i].clamp(min=1e-3)))
        conf = conf.masked_fill(~still, -1)        # only consider masked positions
        idx = conf.topk(k, dim=1).indices          # most confident masked positions
        for b in range(n):
            x[b, idx[b]] = pred[b, idx[b]]
    return [tok.decode(seq, skip_special_tokens=True) for seq in x]
```

This is the confidence-based version; the random-commit version simply replaces
the `topk` selection with a random mask.

---

## 7. Score-based discrete diffusion (SEDD)

The strongest results in the pure-discrete family come from reframing the problem
in terms of *score* rather than likelihood. The reference is **SEDD** (Lou,
Meng, Ermon, 2024, "Discrete Diffusion Modeling by Estimating the Ratios of the
Data Distribution").

The idea: in continuous diffusion, the score is $\nabla_x \log p(x)$. There is no
gradient over a discrete space, but there is a discrete analogue — the **ratio**
of probabilities between neighboring states, $p(y)/p(x)$ for states $y$ that
differ from $x$ by one token. SEDD learns these *concrete score* ratios with a
**denoising score-entropy** loss, a discrete counterpart of score matching.

Why it matters:

- It significantly closed the gap with autoregressive language models in
  perplexity, which the earlier D3PM/multinomial models had not.
- It admits flexible samplers (including remasking and analytic steps) derived
  from the learned ratios.

For a demonstrator SEDD is more involved than masked diffusion and is best read
*after* you have the absorbing model working. But it is the thing to study if you
want to understand why discrete diffusion became competitive rather than a
curiosity. Note that MDLM and the Shi et al. work later showed that a carefully
done *masked* objective recovers much of SEDD's quality with a simpler loss, so
the two threads have partly converged.

---

## 8. Continuous-embedding approaches (the other family)

The alternative to working in discrete space is to embed tokens into a
continuous space and run ordinary Gaussian diffusion there.

**Diffusion-LM** (Li et al., 2022) is the seminal work. The pipeline:

1. Map each token to a continuous embedding vector.
2. Run standard continuous diffusion (forward Gaussian noising, epsilon- or
   x0-prediction denoiser) on the *sequence of embeddings*.
3. Add a "rounding" step that maps denoised embeddings back to discrete tokens,
   with an auxiliary loss to keep embeddings near valid token vectors.

The appeal is that you reuse the entire continuous toolkit you already know —
including classifier and classifier-free guidance, which made Diffusion-LM
attractive for *controllable* generation (steering syntax, length, sentiment).
Successors include **Analog Bits** (Chen et al., 2022, encode tokens as binary
bits treated as real numbers), **SSD-LM** (semi-autoregressive, diffuses over
blocks in the natural vocabulary space), and **CDCD** (Dieleman et al., 2022,
continuous diffusion with learned embeddings and a score-interpolation trick).

The tradeoff: the rounding step is the persistent weak point. Embeddings that
land between valid tokens have to be snapped back, and getting this stable while
matching discrete-diffusion quality proved hard. As of the current landscape,
the strongest perplexity numbers come from the discrete side (SEDD, MDLM), while
the continuous-embedding side remains compelling mainly for fine-grained
controllability. Know this family exists and what it trades; build your
demonstrator in the discrete family.

---

## 9. Building the demonstrator on TinyStories

### Dataset

TinyStories (Eldein and Li, 2023) is a corpus of short stories written with a
small vocabulary a young child would know. It is ideal for a demonstrator: small
vocabulary, short sequences, and coherent output achievable at a tractable
scale. There is an original version and a cleaner `TinyStoriesV2`; either works,
check the dataset card and prefer V2 if available. The corpus has roughly two
million stories — far more than a small model needs, so you will typically train
on a slice (a few hundred thousand stories is plenty) and the binding constraint
will be compute, not data.

### Tokenizer and data

```python
from datasets import load_dataset
from transformers import GPT2TokenizerFast

tok = GPT2TokenizerFast.from_pretrained("gpt2")
tok.add_special_tokens({"mask_token": "[MASK]"})
MASK_ID = tok.mask_token_id
VOCAB   = len(tok)
SEQ_LEN = 128

ds = load_dataset("roneneldan/TinyStories", split="train[:200000]")

def encode(batch):
    out = tok(batch["text"], truncation=True, max_length=SEQ_LEN,
              padding="max_length")
    return {"input_ids": out["input_ids"]}

ds = ds.map(encode, batched=True, remove_columns=["text"])
ds.set_format("torch", columns=["input_ids"])
```

### Schedule and forward corruption

```python
import torch

def mask_prob(t):                 # t in [0,1]; cosine usually samples better
    return 1 - torch.cos(torch.pi * t / 2)

def corrupt(x, t):
    p = mask_prob(t).unsqueeze(1)
    mask = torch.rand_like(x.float()) < p
    x_t = torch.where(mask, MASK_ID, x)
    return x_t, mask
```

### Model: a small bidirectional transformer

The model is an encoder (full bidirectional attention — no causal mask, which is
the whole point) that takes the corrupted tokens plus a time embedding and
outputs logits over the vocabulary at every position.

```python
import torch.nn as nn

class TimeEmbed(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
    def forward(self, t):
        return self.lin(t.unsqueeze(-1))

class DiffLM(nn.Module):
    def __init__(self, vocab, d=384, layers=6, heads=6, seq=SEQ_LEN):
        super().__init__()
        self.tok  = nn.Embedding(vocab, d)
        self.pos  = nn.Embedding(seq, d)
        self.time = TimeEmbed(d)
        enc = nn.TransformerEncoderLayer(d, heads, 4*d, batch_first=True,
                                         activation="gelu", norm_first=True)
        self.blocks = nn.TransformerEncoder(enc, layers)
        self.head   = nn.Linear(d, vocab)
    def forward(self, x, t):
        pos = torch.arange(x.size(1), device=x.device)
        h = self.tok(x) + self.pos(pos)[None] + self.time(t)[:, None]
        h = self.blocks(h)
        return self.head(h)
```

### Loss

```python
import torch.nn.functional as F

def loss_fn(model, x):
    B = x.size(0)
    t = torch.rand(B, device=x.device).clamp(1e-3, 1.0)
    x_t, mask = corrupt(x, t)
    logits = model(x_t, t)
    ce = F.cross_entropy(logits.transpose(1, 2), x, reduction="none")
    w  = 1.0 / mask_prob(t).clamp(min=1e-3)
    per_ex = (ce * mask).sum(1) / mask.sum(1).clamp(min=1)
    return (per_ex * w).mean()
```

### Training loop

```python
from torch.utils.data import DataLoader

def train(model, ds, epochs=3, bs=64, lr=3e-4, device="cuda"):
    model.to(device).train()
    dl  = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    for ep in range(epochs):
        for step, batch in enumerate(dl):
            x = batch["input_ids"].to(device)
            loss = loss_fn(model, x)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 100 == 0:
                print(f"ep {ep} step {step} loss {loss.item():.4f}")

model = DiffLM(VOCAB)
train(model, ds)
```

On a single modern GPU, a model of this size on a couple hundred thousand stories
reaches readable (if slightly wobbly) children's-story samples in a few hours.

### Sampling

Use the confidence-based sampler from section 6. Start with `steps=64`; try
`steps=128` and `steps=16` to feel the speed/quality tradeoff.

---

## 10. Evaluation

Discrete diffusion does not give you an exact log-likelihood the way an
autoregressive model does, but the ELBO gives an upper bound on negative
log-likelihood, which converts to a **bound on perplexity**. Report it on a held-
out split and track it across training; it is your primary quantitative signal.

Complementary checks:

- **Generative perplexity** under a separate, strong autoregressive model:
  generate samples, score them with the external model. Lower is more fluent, but
  beware — this rewards bland, high-probability text, so read it alongside
  diversity.
- **Diversity / distinct-n**: fraction of unique n-grams across samples, to catch
  mode collapse and repetition.
- **Qualitative reading**: for TinyStories, simply reading a handful of samples
  tells you a lot — are they coherent stories, do they stay on topic, is grammar
  intact.

Watch the speed/quality curve as a function of sampling steps; it is one of the
most informative diagnostics for a diffusion LM and a good thing to plot.

---

## 11. Practical failure modes and fixes

- **Dropping the schedule weight $w(t)$.** Without it you have plain BERT, not a
  valid diffusion bound; samples degrade. Keep the $1/\sigma(t)$ reweighting.
- **Accidentally causal attention.** The model must be bidirectional. A causal
  mask silently turns it into a weak autoregressive model and defeats the
  parallel sampler. Use an encoder, no causal mask.
- **Too few sampling steps.** One- or two-step generation looks like noise.
  Increase steps; use confidence-based unmasking before blaming the model.
- **Forgetting to resize embeddings after adding `[MASK]`.** If you load any
  pretrained weights, call `resize_token_embeddings`. Training from scratch with
  `VOCAB = len(tok)` sidesteps this.
- **Time conditioning too weak.** If the model ignores `t`, samples don't improve
  with more steps. Make sure the time embedding actually reaches every block
  (add it to the input, or inject per-block).
- **Padding leaking into the loss.** Mask out padding positions so the model is
  not rewarded for predicting `[PAD]`; otherwise short stories dominate the loss.
- **Evaluating only on generative perplexity.** It rewards blandness; always pair
  with a diversity metric and direct reading.

---

## 12. Where the field is going

A few threads worth tracking, stated at a level that should remain useful as
specifics evolve:

- **Convergence of masked and score-based views.** MDLM/Shi-style masked
  objectives and SEDD-style score objectives have moved toward similar quality,
  and the theory connecting them has tightened. Expect the practical recipe to
  keep simplifying.
- **Speed as the headline advantage.** The parallel, any-order sampler is the
  reason for the interest: a diffusion LM can in principle emit many tokens per
  step instead of one. Work on few-step samplers and distillation targets this
  directly.
- **Controllability.** Bidirectional, any-order generation makes infilling and
  constraint satisfaction natural — you can fix some tokens and diffuse the rest.
  This is a structural advantage over left-to-right models.
- **Scaling.** The open question is how the quality and speed story holds at large
  scale against highly optimized autoregressive models. This is an active and
  fast-moving area, so verify current claims against recent results rather than
  trusting any fixed snapshot.

Because this section is the most time-sensitive, treat it as a pointer to topics
to check rather than settled fact.

---

## 13. Annotated reading list

Read roughly in this order.

1. **D3PM** — Austin et al., 2021, "Structured Denoising Diffusion Models in
   Discrete State-Spaces." The foundation: transition matrices, the general
   framework, uniform vs. absorbing corruption. Start here.
2. **Multinomial Diffusion** — Hoogeboom et al., 2021, "Argmax Flows and
   Multinomial Diffusion." The uniform-corruption case, cleanly developed.
3. **MDLM** — Sahoo et al., 2024, "Simple and Effective Masked Diffusion
   Language Models." The cleanest practical masked objective; closest to your
   demonstrator. Read alongside the concurrent **Shi et al., 2024** masked-
   diffusion paper.
4. **SEDD** — Lou, Meng, Ermon, 2024, "Discrete Diffusion Modeling by
   Estimating the Ratios of the Data Distribution." The score-entropy approach
   that made discrete diffusion competitive. Read after the masked model works.
5. **Diffusion-LM** — Li et al., 2022, "Diffusion-LM Improves Controllable Text
   Generation." The continuous-embedding family and the controllability angle.
6. **CDCD** — Dieleman et al., 2022, "Continuous diffusion for categorical
   data." A thoughtful continuous-embedding treatment worth reading for
   contrast.

These citations are from training knowledge and the field moves quickly —
verify exact titles, authors, and venues, and look for newer follow-ups, before
relying on them.

---

## 14. Exercises

1. **Schedule comparison.** Train two models, one linear and one cosine schedule,
   identical otherwise. Plot held-out perplexity-bound and generative perplexity
   vs. sampling steps for both. Which wins, and where on the steps axis?
2. **Random vs. confidence unmasking.** Hold the trained model fixed and compare
   the two samplers at equal step counts. Quantify with distinct-n and reading.
3. **Step budget curve.** Plot sample quality (generative PPL and a diversity
   metric) against number of sampling steps from 4 to 256. Find the knee.
4. **BERT ablation.** Remove the $1/\sigma(t)$ weight so the loss becomes plain masked
   cross-entropy. Measure how much sample quality drops. This makes the
   diffusion-vs-BERT distinction concrete.
5. **Infilling.** Modify the sampler to keep a user-provided prefix and suffix
   fixed (never masked) and diffuse only the middle. Confirm the model respects
   both sides — the payoff of bidirectionality.
6. **Uniform corruption.** Implement the multinomial variant (random-token
   corruption instead of `[MASK]`) and compare training stability and sample
   quality against the absorbing model on the same data.
7. **(Stretch) Concrete score.** Read SEDD and implement the denoising
   score-entropy loss for a small vocabulary. Compare against your masked model.

---

*This course is intended as a learning scaffold. Code is illustrative and
written for clarity over efficiency; the citations reflect training-time
knowledge of a fast-moving field and should be verified against current
sources.*