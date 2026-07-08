# Polish Byte-Level BPE Tokenizer

## Objective

The goal was to implement and study a byte-level BPE tokenizer for Polish text.
The tokenizer was trained on a balanced Polish corpus, evaluated with aggregate
metrics and targeted stress tests, and saved as `tokenizer.json`.

## Tokenizer Design

The tokenizer uses a byte-level base vocabulary, so every input string can be
encoded and decoded without out-of-vocabulary failures.

Normalization choices:

- Unicode NFC normalization;
- CRLF/CR line endings normalized to `\n`;
- removal of invisible/control junk such as BOM, soft hyphen, and zero-width
  characters;
- no lowercasing;
- no punctuation stripping;
- Polish diacritics preserved.

Pre-tokenization choices used during experimentation:

- letters, digits, punctuation, whitespace, and newlines are handled separately;
- leading spaces before words are attached to the following word;
- digit runs are split from the right into groups of up to three;
- newline is a hard boundary;
- horizontal whitespace runs are preserved so indentation can become learnable.

Final exported tokenizer:

- Hugging Face `tokenizers` JSON format;
- ByteLevel BPE model;
- NFC normalizer;
- ByteLevel pre-tokenizer and decoder;
- 8192 total vocabulary entries, including reserved special tokens.

## Training Data

The first experiments used `Pan Tadeusz` as a small Polish sample. Later runs
used a balanced subset of Polish DynaWord plus a small Polish FineWeb sample.

The final compared corpus combined:

- curated Polish sources: Wikipedia, Wikisource, Wolne Lektury, novels, ELTeC,
  Wikibooks, Wikinews, Wikivoyage, Wikiquote;
- capped legal/parliamentary text;
- contemporary Polish web text from FineWeb.

The combined training/evaluation sample was about 17.2 MB of UTF-8 text, with
FineWeb contributing about 28.5% by bytes.

Legal/parliamentary data was deliberately capped. In the raw DynaWord corpus,
legal/parliamentary sources dominate the token count, which would make the
tokenizer too specialized for legal style if used without balancing.

## Evaluation Metrics

The main metrics were:

- bytes per token: higher means better compression;
- fertility: tokens per word, lower is better;
- vocabulary utilization on held-out text;
- targeted stress tests for Polish spelling, numbers, code, markdown, English,
  emoji, and chat-like delimiters.

## Vocab Size Results

Held-out results:

| Dataset | Vocab | Bytes/Token | Tokens/Word | Vocab Utilization |
|---|---:|---:|---:|---:|
| DynaWord | 512 | 1.882 | 4.178 | 83.17% |
| DynaWord | 1024 | 2.296 | 3.424 | 91.64% |
| DynaWord | 2048 | 2.663 | 2.953 | 95.74% |
| DynaWord | 4096 | 3.046 | 2.581 | 97.38% |
| DynaWord | 8192 | 3.458 | 2.274 | 96.95% |
| DynaWord + FineWeb | 512 | 1.932 | 4.016 | 83.17% |
| DynaWord + FineWeb | 1024 | 2.377 | 3.264 | 91.05% |
| DynaWord + FineWeb | 2048 | 2.771 | 2.801 | 95.25% |
| DynaWord + FineWeb | 4096 | 3.207 | 2.419 | 96.55% |
| DynaWord + FineWeb | 8192 | 3.661 | 2.120 | 95.21% |

The best result in these experiments was the DynaWord + FineWeb tokenizer with vocab
size 8192:

```text
bytes/token = 3.661
tokens/word = 2.120
vocab utilization = 95.21%
```

## Stress Tests

Token counts for selected stress cases:

| Case | DynaWord 2048 | DynaWord 8192 | +FineWeb 2048 | +FineWeb 8192 |
|---|---:|---:|---:|---:|
| Polish diacritics | 31 | 29 | 31 | 29 |
| Polish without diacritics | 30 | 29 | 30 | 28 |
| Large numbers | 38 | 33 | 38 | 33 |
| Code indentation | 46 | 43 | 45 | 40 |
| Markdown | 41 | 34 | 38 | 33 |
| Mixed Polish/English | 38 | 30 | 38 | 30 |
| Emoji and symbols | 43 | 40 | 44 | 39 |
| Chat-like text | 41 | 36 | 42 | 36 |

The larger vocabulary improved nearly all stress cases. FineWeb helped the
aggregate metrics and slightly improved web-like/code/markdown examples, but the
small FineWeb sample did not dramatically change every targeted case.

## Findings

1. Byte-level BPE gives reliable round-trip behavior.

   Even out-of-domain text, emoji, and rare symbols can be encoded and decoded.
   The cost is that rare inputs may fragment into byte-level pieces.

2. Pre-tokenization strongly shapes what BPE can learn.

   Leading-space word tokens, newline boundaries, and right-grouped numbers all
   appear directly in the learned behavior.

3. Polish-specific chunks emerge naturally.

   Frequent Polish characters and sequences such as `ł`, `ą`, `ę`, `rz`, `cz`,
   `sz`, `ch`, `ię`, and `ści` are learned without linguistic rules.

4. Data mix matters.

   A literary-only tokenizer works for Polish prose but fragments numbers, code,
   markdown, English, emoji, and chat-like text. Adding more diverse data
   improves compression and robustness.

5. Vocab size matters substantially.

   Moving from 2048 to 8192 improved both aggregate metrics and stress tests.
   At 8192, vocabulary utilization was still high, so the vocab was not
   obviously too large for the sampled data.

6. Legal text should be capped.

   Legal/parliamentary text is useful but can dominate Polish corpora. Keeping it
   capped produced a more balanced tokenizer training mix.

## Limitations

- Special tokens are reserved in the vocabulary, but explicit special-token
  encoding policy still needs to be added.
- The corpus sample is small compared with production tokenizer training data.
- Numbers, emoji, and chat delimiters still fragment more than desired.
- More code/markdown/web data would likely improve non-prose behavior.

## Final Artifact

The selected tokenizer for this checkpoint is:

```text
tokenizer.json
```

It is a Hugging Face-compatible ByteLevel BPE tokenizer exported from the
DynaWord + FineWeb 8192-vocab run, which had the best held-out compression and
fertility among the tested configurations.
