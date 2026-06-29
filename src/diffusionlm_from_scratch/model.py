from dataclasses import dataclass

import math 

import torch
from torch import nn
from torch.nn import functional as F

from timm.models.vision_transformer import Attention, Mlp



def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, time_scale=1000.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        # Continuous-time t lives in [0, 1]; the sinusoidal embedding was designed
        # for integer diffusion steps (~0..1000). Without rescaling, the embedding
        # barely varies across t and time conditioning is too weak (RESEARCH.md s11).
        self.time_scale = time_scale

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t * self.time_scale, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb



class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of the DiT: adaLN modulation followed by a projection to
    vocabulary logits (the discrete-diffusion analogue of projecting back to
    image patches).
    """
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, vocab_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


@dataclass
class DiTConfig:
    vocab_size: int = 50258      # GPT-2 vocab (50257) + 1 for [MASK]
    max_seq_len: int = 128
    hidden_size: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0


class DiT(nn.Module):
    """
    Diffusion Transformer for discrete (absorbing-state / masked) text diffusion.

    Unlike the image DiT, the input is a sequence of (possibly masked) token ids
    rather than a latent patch grid, and the output is per-position logits over
    the vocabulary, i.e. the x0-prediction p_theta(x_0 | x_t, t). Attention is
    fully bidirectional (no causal mask) -- that is the whole point.
    """
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads

        self.tok_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_len, config.hidden_size))
        self.t_embedder = TimestepEmbedder(config.hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(config.hidden_size, config.num_heads, mlp_ratio=config.mlp_ratio)
            for _ in range(config.depth)
        ])
        self.final_layer = FinalLayer(config.hidden_size, config.vocab_size)

        self.initialize_weights()

    def initialize_weights(self):
        # Standard transformer init for linears.
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Token + positional embeddings: normal init.
        nn.init.normal_(self.tok_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

        # Timestep embedder MLP.
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out the adaLN modulation layers (adaLN-Zero): every block starts
        # as an identity, which is what makes deep DiTs train stably.
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out the final layer so initial predictions are unbiased.
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t):
        """
        x: (B, L) long tensor of (possibly masked) token ids.
        t: (B,) timestep, either a float in [0, 1] (continuous-time) or an
           integer index; it is fed through the sinusoidal TimestepEmbedder.
        returns: (B, L, vocab_size) logits = p_theta(x_0 | x_t, t).
        """
        L = x.size(1)
        h = self.tok_embed(x) + self.pos_embed[:, :L]   # (B, L, D)
        c = self.t_embedder(t)                           # (B, D)
        for block in self.blocks:
            h = block(h, c)                              # bidirectional attention
        logits = self.final_layer(h, c)                  # (B, L, vocab_size)
        return logits


def DiT_small(vocab_size, max_seq_len=128, **kwargs):
    return DiT(DiTConfig(vocab_size=vocab_size, max_seq_len=max_seq_len,
                         hidden_size=384, depth=12, num_heads=6, **kwargs))


def DiT_base(vocab_size, max_seq_len=128, **kwargs):
    return DiT(DiTConfig(vocab_size=vocab_size, max_seq_len=max_seq_len,
                         hidden_size=768, depth=12, num_heads=12, **kwargs))

