# Building a BPE Tokenizer from Scratch

A hands-on workshop notebook ([tokenizer_workshop_explanation.ipynb](tokenizer_workshop_explanation.ipynb))
that implements **Byte-Pair Encoding (BPE)** — the tokenization algorithm behind modern LLMs — in
plain Python, with no external ML libraries. Every concept is followed by a runnable example on a
real Polish literary corpus.

## Training a BPE tokenizer (the core)

Training learns a set of **merge rules** from a corpus. The whole algorithm is four small,
independently testable functions:

| Function | Role |
|-------------------------------|-------------------------------------------------------------------------------|
| `get_pair_counts(ids)`        | Count every adjacent token pair.                                              |
| `merge(ids, pair, new_id)`    | Rewrite the sequence, replacing one pair with a new token ID.                 |
| `train_bpe(text, vocab_size)` | The training loop that ties them together.                                    |
| `encode` / `decode`           | Use the trained rules on new text (not training, but shown for completeness). |

### The training loop

```python
def train_bpe(text, vocab_size):
    token_ids = list(text.encode("utf-8"))  # start from raw bytes
    merges = {}                              # (a, b) -> new_id, in learning order
    next_id = 256                            # 0–255 are the byte values
    for _ in range(vocab_size - 256):
        counts = get_pair_counts(token_ids)
        if not counts:
            break
        best_pair = max(counts, key=counts.get)   # most frequent pair
        token_ids = merge(token_ids, best_pair, next_id)
        merges[best_pair] = next_id
        next_id += 1
    return token_ids, merges
```

Key ideas:

- **Start from bytes, not characters.** `text.encode("utf-8")` gives a base alphabet of exactly
  256 tokens, so *any* text is representable and there are no "unknown character" gaps. Polish
  letters like `ł`, `ę`, `ó` are 2 bytes each — BPE learns to glue them back into single tokens.
- **Always merge the most frequent pair** — that gives the biggest reduction in sequence length
  per merge.
- **Order matters.** Merges are recorded in the order they were learned, because later merges are
  built on top of earlier ones (e.g. `nie` = `n` + the previously-learned `ie` token). The encoder
  must replay them in that same order.
- **`N` merges = `vocab_size − 256`.** The final vocabulary is the 256 base bytes plus one new
  token per merge.

### Training run on a real corpus

The notebook trains on a Polish literary text (`krzyzacy_tom_1.txt`) with `NUM_OF_MERGES = 1000`
(target vocab size 1256):

```
Trained 1000 merges (vocab size 1256).
Corpus: 732,128 bytes -> 245,803 tokens (66.4% reduction).
```

The learned merges recover meaningful Polish subword pieces — `ie`, `nie`, `cz`, `rz`, `ch`,
`się`, `dzie`, suffixes like `ał`, `ość`, etc. — confirming BPE discovers real linguistic units
purely from frequency statistics.

### Saving / shipping the trained tokenizer

The trained tokenizer *is* just its merge rules. They're exported to
`silaczka_tokenizer_merges.json`, with each `(a, b) -> new_id` rule flattened to `[a, b, new_id]`
(tuple keys aren't valid JSON). JSON list order preserves the learning order the encoder depends
on. A round-trip assert confirms the reloaded rules match the originals.

## Using the trained tokenizer

- **`encode(text, merges)`** — text → token IDs. Applies merges in learned order (earliest-learned
  matching pair first, via `min(..., key=merges.get)`), *not* by frequency in the new text.
- **`build_vocab(merges)`** — reconstructs an `id -> bytes` lookup by walking the merges in order.
- **`decode(ids, vocab)`** — token IDs → text. Joins all token bytes first, then decodes UTF-8
  **once** at the end, so multi-byte characters split across tokens reassemble correctly.

Correctness is proven by the round-trip property `decode(encode(text)) == text`, verified on
sentences that were never in the training corpus.

## Running it

Open [tokenizer_workshop_explanation.ipynb](tokenizer_workshop_explanation.ipynb) and run the cells
top to bottom. Requirements: Python 3 and a Jupyter environment — only the standard library
(`re`, `json`) is used. A UTF-8 text file is needed as the training corpus (the notebook expects
`krzyzacy_tom_1.txt` in the repo root).
