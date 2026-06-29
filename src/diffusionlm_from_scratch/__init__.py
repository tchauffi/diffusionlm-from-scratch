"""Masked (absorbing-state) diffusion language model, from scratch.

Quick inference::

    from diffusionlm_from_scratch import DiffusionLM

    lm = DiffusionLM.from_pretrained("tchauffi/diffusionlm-from-scratch")
    for story in lm.generate(n=4, seq_len=80, temperature=0.9):
        print(story)
"""

from .model import DiT, DiTConfig
from .inference import DiffusionLM, sample, refine

__all__ = ["DiffusionLM", "DiT", "DiTConfig", "sample", "refine", "main"]


def main() -> None:
    print("Hello from diffusionlm-from-scratch!")
