from pathlib import Path

__SENSITIVE_WORDS_PATH__ = Path(__file__).parent / "sensitive_words.txt"


def load_sensitive_words() -> list[str]:
    file_lines = __SENSITIVE_WORDS_PATH__.read_text(encoding="utf-8").splitlines()

    return [
        line.strip() for line in file_lines if line.strip() and not line.startswith("#")
    ]


def sensitive_words_filter(text: str) -> str:
    sensitive_words = load_sensitive_words()
    for word in sensitive_words:
        if word in text:
            return False, "输出规则命中敏感词"
    return True, "输出规则未命中敏感词"
