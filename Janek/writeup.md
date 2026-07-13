# Polish Byte-Level BPE Tokenizer

## Overview

The final artifact is a Hugging Face-compatible ByteLevel BPE tokenizer for
Polish text:

```text
tokenizer.json
```

It was trained on a balanced DynaWord + FineWeb Polish corpus with an 8192-entry
vocabulary. This was the largest vocabulary I trained locally within practical
compute limits.

The strongest held-out result among the tested configurations was:

```text
bytes/token = 3.661
tokens/word = 2.120
vocab utilization = 95.21%
```

## Research Question And Compute Constraints

The goal was to build a robust byte-level BPE tokenizer for Polish text and
understand which choices mattered most under local compute constraints.

The main questions were:

- how much vocabulary size improves Polish compression;
- whether adding contemporary web text improves robustness over DynaWord alone;
- what kinds of tokens are actually learned;
- how the final tokenizer compares with nearby peer baselines.

The final vocabulary size, 8192, should be read as a local-compute checkpoint,
not as a claim that larger vocabularies would stop improving. The experiments
show that 8192 was still useful and highly utilized, so larger vocabularies
would be a natural next experiment.

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

The exported tokenizer uses:

- Hugging Face `tokenizers` JSON format;
- ByteLevel BPE model;
- NFC normalizer;
- ByteLevel pre-tokenizer and decoder;
- 8192 total vocabulary entries, including reserved special tokens.

During experimentation I also used a custom pre-tokenization policy that kept
letters, digits, punctuation, whitespace, and newlines separate; attached leading
spaces to following words; split digit runs from the right into groups of up to
three; and treated newline as a hard boundary.

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

## Evaluation Protocol

The main metrics were:

- bytes per token: higher means better compression;
- fertility: tokens per word, lower is better;
- vocabulary utilization on held-out text;
- round-trip correctness;
- targeted stress tests for Polish spelling, numbers, code, markdown, English,
  emoji, and chat-like delimiters.

## Ablation Study

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

Findings from the ablation:

1. Vocabulary size was the strongest lever. Moving from 512 to 8192 reduced
   held-out fertility from 4.016 to 2.120 on the DynaWord + FineWeb mix.
2. Adding FineWeb improved the held-out result at every tested vocabulary size.
   At 8192, fertility improved from 2.274 to 2.120.
3. Vocabulary utilization stayed high at 8192, so the vocabulary was not
   obviously too large for the sampled data.
4. Larger vocabularies were not tested because local training cost became the
   limiting factor.

## Baseline Comparison

I compared the final tokenizer against nearby local peer artifacts on the same
small evaluation suite. This is not a full benchmark, but it gives context for
the selected artifact.

| Tokenizer | Vocab | Total Tokens On Suite | Avg Bytes/Token | Avg Tokens/Word |
|---|---:|---:|---:|---:|
| Janek 8192 | 8192 | 245 | 2.477 | 5.571 |
| Kuba 14000 | 14000 | 241 | 2.494 | 5.395 |
| Arek 8192 | 8192 | 270 | 2.273 | 6.387 |
| Arek 15000 | 15000 | 256 | 2.423 | 5.982 |

Selected case results:

| Case | Janek 8192 | Kuba 14000 | Arek 8192 | Arek 15000 |
|---|---:|---:|---:|---:|
| Pan Tadeusz opening | 15 | 19 | 15 | 13 |
| Modern web Polish | 23 | 22 | 27 | 26 |
| Legal Polish | 17 | 16 | 18 | 17 |
| Code/Markdown | 48 | 49 | 50 | 49 |
| Chat-like text | 32 | 31 | 33 | 31 |
| Mixed Polish/English | 35 | 32 | 40 | 38 |
| Emoji/symbols | 36 | 36 | 38 | 37 |
| Numbers | 39 | 36 | 49 | 45 |

The comparison suggests that the 8192-vocab tokenizer is competitive with
larger local baselines on this small suite, especially for Polish prose and
code/markdown. Kuba's larger Wikipedia-trained tokenizer is slightly better on
web/legal/mixed English and numbers. Arek's larger 15000 vocabulary improves over
his 8192 variant, which matches the broader vocabulary-size trend.

## Token Quality Analysis

Aggregate compression does not show what BPE actually learned, so I inspected
long tokens, frequent held-out tokens, morphology probes, and failure cases.

Longest decoded vocabulary entries included:

- ` Rzeczypospolitej`
- ` recapitalisation`
- `................`
- ` najważniejszych`
- ` rozporządzenia`
- ` bezpieczeństwa`
- ` prawdopodobnie`
- ` przewodniczący`
- ` poszczególnych`
- ` infrastruktury`

Frequent held-out tokens included punctuation and useful Polish leading-space
tokens:

- `,`
- `\n`
- `.`
- ` `
- ` w`
- ` i`
- ` z`
- ` na`
- ` się`
- ` do`

Polish morphology probes:

| Text | Tokens | Split |
|---|---:|---|
| `odpowiedzialność` | 5 | `od | powie | dzi | al | ność` |
| `odpowiedzialności` | 5 | `od | powie | dzi | al | ności` |
| `niepodległość` | 4 | `nie | pod | leg | łość` |
| `niepodległościowy` | 5 | `nie | pod | leg | łości | owy` |
| `najprawdopodobniej` | 4 | `naj | praw | dopodob | niej` |
| `przedsiębiorstwo` | 5 | `prze | d | się | bior | stwo` |
| `Rzeczpospolita` | 5 | `R | ze | cz | pospoli | ta` |
| `dziewięćdziesięciokilkuletni` | 10 | `dzie | wię | ć | dzie | się | cio | ki | l | ku | letni` |

This shows that the tokenizer learns useful Polish chunks, but BPE does not
consistently preserve linguistic morpheme boundaries. For example,
`odpowiedzialność` and `odpowiedzialności` share most of their prefix split, but
the suffix appears as `ność` vs `ności`; `przedsiębiorstwo` is split less
cleanly.

Stress probes:

| Text Type | Tokens | Split |
|---|---:|---|
| Date `2026-07-08` | 7 | `2 | 02 | 6 | - | 07 | - | 08` |
| Large number `1_000_000` | 5 | `1 | _ | 000 | _ | 000` |
| Decimal `3,14159` | 5 | `3 | , | 14 | 15 | 9` |
| Python signature | 22 | fragments identifiers and punctuation |
| Markdown heading/list | 10 | keeps markdown punctuation mostly separate |
| Chat delimiters | 20 | delimiters fragment heavily |
| Emoji sequence | 15 | falls back to byte-level pieces |
| Mixed scripts | 34 | non-Latin scripts fragment heavily |

The byte-level design preserves round-trip behavior, but non-Polish scripts,
emoji, chat delimiters, and long numeric patterns remain weak spots.

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

2. Vocab size matters substantially.

   Moving from 2048 to 8192 improved both aggregate metrics and stress tests.
   At 8192, vocabulary utilization was still high, so larger vocabularies remain
   worth testing.

3. Data mix matters.

   Adding FineWeb improved held-out compression and robustness over DynaWord
   alone, especially on the final 8192-vocab run.

4. Polish-specific chunks emerge naturally.

   Frequent Polish forms such as `się`, `ność`, `łości`, `prze`, and common
   leading-space function words were learned without explicit linguistic rules.

5. Compression is not the same as morphology.

   The tokenizer learns useful subwords, but it does not consistently align with
   Polish morphemes.

6. Legal text should be capped.

   Legal/parliamentary text is useful but can dominate Polish corpora. Keeping it
   capped produced a more balanced tokenizer training mix.

## Limitations

- Special tokens are reserved in the vocabulary, but explicit special-token
  encoding policy still needs to be added.
- The corpus sample is small compared with production tokenizer training data.
- The final 8192 vocabulary was limited by local compute, not by a proven optimum.
- Numbers, emoji, non-Latin scripts, and chat delimiters still fragment more than
  desired.
- More code/markdown/web/chat data would likely improve non-prose behavior.
- Baseline comparison used a small local suite, not a large shared benchmark.

## Final Artifact

The selected tokenizer for this checkpoint is:

```text
tokenizer.json
```

It is a Hugging Face-compatible ByteLevel BPE tokenizer exported from the
DynaWord + FineWeb 8192-vocab run, which had the best held-out compression and
fertility among the tested configurations.
