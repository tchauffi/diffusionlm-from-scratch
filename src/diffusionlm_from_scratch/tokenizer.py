"""Train a small TinyStories-specialized BPE tokenizer.

GPT-2's 50k vocab is wasteful on a small-vocabulary children's corpus: the token
embedding and the output head together account for most of the model's
parameters, and stories tokenize into more pieces than necessary. A compact BPE
trained on TinyStories shrinks the softmax, lets ``seq_len`` cover whole stories,
and shifts capacity toward the transformer body.

The trained tokenizer keeps the same interface the rest of the pipeline expects:
a ``[MASK]`` absorbing state, a ``[PAD]`` token, and ``mask_token_id`` / ``len``.
"""

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast

from datasets import load_dataset

PAD, UNK, MASK, EOS = "[PAD]", "[UNK]", "[MASK]", "<|endoftext|>"


def train_tinystories_tokenizer(save_path, vocab_size=8192, n_stories=100000,
                                dataset_name="noanabeshima/TinyStoriesV2"):
    """Train a byte-level BPE on TinyStories text and save it to ``save_path``.

    Returns the wrapped :class:`PreTrainedTokenizerFast`.
    """
    ds = load_dataset(dataset_name, split=f"train[:{n_stories}]")

    def text_iter(batch_size=1000):
        for i in range(0, len(ds), batch_size):
            yield ds[i:i + batch_size]["text"]

    tok = Tokenizer(models.BPE(unk_token=UNK))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[PAD, UNK, MASK, EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(text_iter(), trainer=trainer, length=len(ds))

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token=PAD, unk_token=UNK, mask_token=MASK, eos_token=EOS,
    )
    fast.save_pretrained(save_path)
    return fast


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "tinystories_tokenizer"
    t = train_tinystories_tokenizer(path)
    print(f"vocab_size={len(t)} mask_id={t.mask_token_id} saved to {path}")
