"""Absorbing-state (masked) noise schedule and forward corruption.

This is the continuous-time masked-diffusion process from RESEARCH.md (sections
4-6). A timestep ``t`` in [0, 1] sets a *cumulative masking probability*
``sigma(t)`` (``sigma(0)=0``, ``sigma(1)=1``); each token is independently
replaced by ``[MASK]`` with that probability. The training loss is a masked
cross-entropy reweighted by ``w(t) = 1/sigma(t)`` -- the term that makes this a
valid diffusion bound rather than plain BERT.
"""

import math
from dataclasses import dataclass

import torch


@dataclass
class AbsorbingScheduler:
    """Continuous-time absorbing-state schedule.

    Args:
        mask_id: the ``[MASK]`` token id (the absorbing state).
        schedule: ``"cosine"`` (default, usually samples better) or ``"linear"``.
        eps: floor on ``sigma(t)`` to keep the ``1/sigma`` weight finite.
    """
    mask_id: int
    schedule: str = "cosine"
    eps: float = 1e-3
    max_weight: float = 10.0   # cap on 1/sigma(t); tames gradient variance at small t
    weight_mode: str = "inv_sigma"   # "inv_sigma" | "uniform" | "sigma"

    def mask_prob(self, t):
        """Cumulative masking probability sigma(t) for t in [0, 1]."""
        if self.schedule == "linear":
            sigma = t
        elif self.schedule == "cosine":
            sigma = 1.0 - torch.cos(math.pi * t / 2.0)
        else:
            raise ValueError(f"unknown schedule: {self.schedule!r}")
        return sigma.clamp(self.eps, 1.0)

    def t_from_mask_prob(self, sigma):
        """Inverse of :meth:`mask_prob`: the t whose masking level is ``sigma``.

        Sampling must keep ``(x_t, t)`` in-distribution -- the model was only ever
        trained on inputs whose masked fraction equals ``sigma(t)``. A reveal
        schedule that conditions on a *linear* t while the actual masked fraction
        follows a different curve feeds the model OOD pairs. Conditioning instead
        on ``t = sigma^{-1}(actual_masked_fraction)`` fixes that for any reveal
        order/schedule.
        """
        sigma = sigma.clamp(self.eps, 1.0) if torch.is_tensor(sigma) \
            else torch.as_tensor(sigma).clamp(self.eps, 1.0)
        if self.schedule == "linear":
            return sigma
        # cosine: sigma = 1 - cos(pi t / 2)  ->  t = (2/pi) * arccos(1 - sigma)
        return (2.0 / math.pi) * torch.arccos((1.0 - sigma).clamp(-1.0, 1.0))

    def loss_weight(self, t):
        """Per-t loss reweighting w(t).

        - ``inv_sigma`` (the MDLM-style ELBO weight): ``min(1/sigma(t), max_weight)``.
          The raw 1/sigma blows up as t -> 0, so a few nearly-unmasked examples
          dominate each batch; clamping bounds that variance. This emphasizes the
          *low*-mask (easy) regime.
        - ``uniform``: ``w(t) = 1``. Equal weight across masking levels -- shifts
          emphasis toward the high-mask regime that generation actually starts in
          (no longer a valid ELBO; trades likelihood-correctness for samples).
        - ``sigma``: ``w(t) = sigma(t)``. Actively up-weights the high-mask regime.
        """
        sigma = self.mask_prob(t)
        if self.weight_mode == "inv_sigma":
            return (1.0 / sigma).clamp(max=self.max_weight)
        if self.weight_mode == "uniform":
            return torch.ones_like(sigma)
        if self.weight_mode == "sigma":
            return sigma
        raise ValueError(f"unknown weight_mode: {self.weight_mode!r}")

    def sample_t(self, batch_size, device):
        """Sample t ~ Uniform[eps, 1] for a minibatch."""
        return torch.rand(batch_size, device=device).clamp(self.eps, 1.0)

    def corrupt(self, x, t, attention_mask=None):
        """Forward process: replace tokens with ``[MASK]`` per the schedule.

        Args:
            x: (B, L) clean token ids.
            t: (B,) timesteps in [0, 1].
            attention_mask: optional (B, L); where 0, the position is padding
                and is never corrupted (and should be excluded from the loss).

        Returns:
            x_t: (B, L) corrupted token ids.
            mask: (B, L) bool, True at positions that were masked (the only
                positions that contribute to the loss).
        """
        p = self.mask_prob(t).unsqueeze(1)               # (B, 1)
        mask = torch.rand(x.shape, device=x.device) < p  # (B, L)
        if attention_mask is not None:
            mask = mask & attention_mask.bool()          # don't corrupt padding
        x_t = torch.where(mask, self.mask_id, x)
        return x_t, mask
