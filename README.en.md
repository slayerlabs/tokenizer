# tokenizer

*[🇵🇱 polski](README.md) · 🇬🇧 English*

Tokenizers from the **[Slayer](https://slayer.fabryka.ai/)** workshop — the open
laboratory of [Fabryka AI](https://fabryka.ai/).

> **Budujemy polskie modele. Otwieramy cały proces.**
> *(We build Polish models. We open up the whole process.)*

Slayer brings together people, compute and method: it trains models, prepares
datasets and publishes measurements that can be checked. Work runs along three
tracks — **models**, **data**, **measurement**; artifacts land on
[HF: SlayerLab](https://huggingface.co/SlayerLab).

This repo comes from the workshop *"How to build a tokenizer for an LLM"* —
byte-level BPE from scratch, in plain Python. The tokenizer is the first decision
in a pretraining pipeline, and one of the few you cannot undo without retraining
the model from zero.

---

## Why tokenize at all

A model doesn't read text character by character, for three reasons:

1. **Sequence length** — the context window is finite. Characters as units mean
   4–5× longer sequences, so 4–5× the cost.
2. **Words don't work either** — the vocabulary explodes, especially in an
   inflected language. `dom · domu · domowi · domem · domach · domowy · domowego`
   is one concept and a dozen IDs. A new form = `<UNK>` = information lost.
3. **Subwords are the compromise** — units between a character and a word.

> A tokenizer is **lossy compression at the level of representation, not
> content**. It learns which byte sequences co-occur often enough to be worth a
> single identifier.

## BPE in four steps

1. Start from the smallest units (bytes).
2. Count the frequency of every **adjacent pair**.
3. Merge the most frequent pair into a new token, record the rule (a merge).
4. Repeat N times, where `N = target vocab − base alphabet`.

On the corpus `aaabdaaabac`:

| pair | merge | corpus |
|---|---|---|
| `aa` → `Z` | 1 | `ZabdZabac` |
| `ab` → `Y` | 2 | `ZYdZYac` |
| `ZY` → `X` | 3 | `XdXac` |

Three merges, vocabulary `{a, b, c, d, Z, Y, X}`.

**Training BPE = learning a vocabulary and a set of merge rules. Tokenizing =
applying those rules in the same order they were created.** The second half is
where people usually go wrong — merges are not applied "most frequent first",
but earliest-learned first.

## Why bytes, not characters

- A Unicode character alphabet means ~150k code points to cover (or `<UNK>` for
  everything missing — emoji, rare scripts).
- UTF-8 has **always 256** byte values. Any text, in any language or script,
  decomposes into bytes **with no exceptions and no `<UNK>`**.
- The cost: a non-ASCII character takes 2–4 bytes instead of one. But BPE
  quickly merges common sequences back — **if** they are common in the training
  corpus.

That last "if" is the crux: a tokenizer trained mostly on English won't give
Polish characters their own tokens, so they fall apart into raw bytes.

## Why this hurts in Polish

The same sentence in Polish typically costs **1.5–3× more tokens** than in
English. The consequences are countable: **cost** (you pay per token),
**context** (the same window holds less) and **quality** (meaning is harder to
assemble from smaller crumbs).

But vocabulary size is a misleading metric. Bielik v3 uses the APT4 tokenizer
with **the same 32,000 tokens as Mistral** — only trained on Polish:

| tokenizer | chars/token | tokens (Polish Constitution) |
|---|---:|---:|
| Mistral (generic) | 2.40 | 747 |
| APT4 (Polish) | **4.78** | **375** |

Half the tokens on the same text, **without a bigger vocabulary**. What counts
is **fertility** (tokens per word for *your* language), not vocabulary size.

## Production pitfalls

- **Special tokens** (`<|endoftext|>`, chat roles) don't come out of BPE — you
  add them by hand with reserved IDs, and they must stay atomic.
- **Vocabulary size is a hyperparameter.** Bigger = shorter sequences, but a
  bigger embedding matrix and tokens seen more rarely.
- **The tokenizer is a separate artifact**, versioned independently of the
  weights. Swap it and the model produces nonsense **without raising an error**
  (embeddings are indexed by ID).
- **Pretokenization before BPE** — production tokenizers split text with a regex
  on word boundaries; they rarely train on a raw byte stream.

---

## The task

Byte-level BPE **from scratch** (no `tokenizers`, no `sentencepiece`), trained on
a Polish corpus and measured on a held-out split: `get_pair_counts` → `merge` →
training loop → `encode` / `decode`, plus a comparison against a generic
tokenizer (`cl100k_base`, `o200k_base`).

## Layout, and how to add yours

**One directory per person** (`Arek/`, `ola/`, `ppuzio/`, …), holding the
tokenizer artifact, the code and a write-up. No imposed structure — it just has
to be reproducible. Fork the repo, add your directory, open a PR against `main`.

A good write-up answers four questions: **what you trained on** (corpus and its
size), **how you measured** (held-out split, definition of a word), **what came
out** (fertility ↓, chars/token ↑, round-trip) and **what surprised you**. Two
notes:

- **Compare against something.** A number on its own says nothing — only next to
  `cl100k_base` on the same text can you see what training on Polish buys.
- **A negative result is still a result.** If your hypothesis didn't hold, say so
  plainly.

## References

- [Slayer](https://slayer.fabryka.ai/) · [Fabryka AI](https://fabryka.ai/) · [HF: SlayerLab](https://huggingface.co/SlayerLab)
- Karpathy, *"Let's build the GPT Tokenizer"* · [minbpe](https://github.com/karpathy/minbpe)
