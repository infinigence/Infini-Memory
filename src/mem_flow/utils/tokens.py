"""Small tokenizer-independent estimate used only for CURRENT rotation."""

from __future__ import annotations

import re


def estimate_tokens(text: str) -> int:
    """Estimate CJK characters and non-CJK word/punctuation token groups."""

    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    without_cjk = re.sub(r"[\u3400-\u9fff]", " ", text)
    other = len(re.findall(r"\w+|[^\w\s]", without_cjk, flags=re.UNICODE))
    return cjk + other


__all__ = ["estimate_tokens"]
