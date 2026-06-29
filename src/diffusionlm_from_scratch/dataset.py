"""TinyStories data pipeline for masked (absorbing-state) text diffusion.

The model is trained to reconstruct clean tokens from a partially-[MASK]ed
sequence (see RESEARCH.md sections 3, 5, 9). This module handles everything up
to the corruption step: building a tokenizer with a dedicated ``[MASK]`` token,
loading TinyStories, and tokenizing it into fixed-length ``input_ids`` plus an
``attention_mask`` that marks the real (non-padding) positions so padding can be
kept out of the loss.
"""

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import GPT2TokenizerFast


@dataclass
class DataConfig:
    dataset_name: str = "noanabeshima/TinyStoriesV2"
    split: str = "train[:200000]"
    val_split: str = "validation[:2000]"
    seq_len: int = 128
    num_proc: int = 4
    tokenizer_path: str = None   # None -> GPT-2; else a saved specialized tokenizer


def build_tokenizer(tokenizer_path=None):
    """Build the tokenizer with an absorbing ``[MASK]`` state.

    ``tokenizer_path=None`` -> GPT-2 (50k vocab). Otherwise load a specialized
    tokenizer saved by :func:`diffusionlm_from_scratch.tokenizer.train_tinystories_tokenizer`
    (small vocab, ``[MASK]``/``[PAD]`` already present).

    Real positions are tracked via the ``attention_mask`` from
    :func:`tokenize_dataset`, so reusing EOS for padding is harmless for the loss.
    Returns ``(tokenizer, mask_id, vocab_size)`` -- pass ``vocab_size`` straight
    into ``DiTConfig``.
    """
    if tokenizer_path is not None:
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok, tok.mask_token_id, len(tok)

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.add_special_tokens({"mask_token": "[MASK]"})
    tok.pad_token = tok.eos_token  # padding is masked out of the loss anyway
    return tok, tok.mask_token_id, len(tok)


def tokenize_dataset(tokenizer, config: DataConfig = DataConfig()):
    """Load TinyStories and tokenize to fixed-length, torch-formatted tensors.

    Each example becomes ``input_ids`` (B, seq_len) and ``attention_mask``
    (1 for real tokens, 0 for padding).
    """
    ds = load_dataset(config.dataset_name, split=config.split)

    def encode(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=config.seq_len,
            padding="max_length",
        )
        return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"]}

    ds = ds.map(
        encode,
        batched=True,
        remove_columns=ds.column_names,
        num_proc=config.num_proc,
    )
    ds.set_format("torch", columns=["input_ids", "attention_mask"])
    return ds


def get_dataloader(config: DataConfig = DataConfig(), batch_size=64, shuffle=True,
                   tokenizer=None, num_workers=2):
    """Convenience: build tokenizer (if needed), tokenize, and wrap in a DataLoader.

    Returns ``(dataloader, tokenizer, mask_id, vocab_size)``.
    """
    if tokenizer is None:
        tokenizer, mask_id, vocab_size = build_tokenizer(config.tokenizer_path)
    else:
        mask_id, vocab_size = tokenizer.mask_token_id, len(tokenizer)

    ds = tokenize_dataset(tokenizer, config)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dl, tokenizer, mask_id, vocab_size
