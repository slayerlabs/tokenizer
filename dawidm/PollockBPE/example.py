from pathlib import Path

from tokenizers import Tokenizer


TOKENIZER_PATH = Path(__file__).with_name("tokenizer.json")


def main() -> None:
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    text = "Pollock is a small language model."
    token_ids = tokenizer.encode(text).ids
    decoded = tokenizer.decode(token_ids)

    assert decoded == text
    print(token_ids)
    print(decoded)


if __name__ == "__main__":
    main()
