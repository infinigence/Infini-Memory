"""Small deterministic BM25 helpers shared by retrieval strategies."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

from ..models import MemoryDocument
from ..utils.codec import strip_yaml_front_matter


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u3400-\u9fff]")
_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def tokenize(value: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(value)]


def bm25_rank(
    query: str,
    texts: Iterable[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[int, float]]:
    documents = [tokenize(text) for text in texts]
    terms = tokenize(query)
    if not documents or not terms:
        return []
    average_length = sum(map(len, documents)) / len(documents) or 1.0
    frequencies = {
        term: sum(1 for document in documents if term in set(document))
        for term in set(terms)
    }
    ranked: list[tuple[int, float]] = []
    for index, document in enumerate(documents):
        counts = Counter(document)
        score = 0.0
        for term in terms:
            frequency = counts[term]
            if not frequency:
                continue
            inverse = math.log(
                1
                + (len(documents) - frequencies[term] + 0.5)
                / (frequencies[term] + 0.5)
            )
            denominator = frequency + k1 * (
                1 - b + b * len(document) / average_length
            )
            score += inverse * frequency * (k1 + 1) / denominator
        if score > 0:
            ranked.append((index, score))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))


def split_h1(markdown: str) -> list[tuple[str, str]]:
    body = strip_yaml_front_matter(markdown)
    matches = list(_H1_RE.finditer(body))
    if not matches:
        return [("", body)]
    partitions: list[tuple[str, str]] = []
    preamble = body[: matches[0].start()].strip()
    if preamble:
        partitions.append(("", preamble))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        partitions.append((match.group(1).strip(), body[match.start() : end].strip()))
    return partitions


def bm25_documents(
    query: str,
    documents: list[MemoryDocument],
    limit: int,
    *,
    k1: float,
    b: float,
) -> list[MemoryDocument]:
    ranks = bm25_rank(
        query,
        [strip_yaml_front_matter(item.content) for item in documents],
        k1=k1,
        b=b,
    )
    return [documents[index] for index, _ in ranks[:limit]]


def bm25_partitions(
    query: str,
    documents: list[MemoryDocument],
    limit: int,
    *,
    k1: float,
    b: float,
) -> list[MemoryDocument]:
    partitions: list[tuple[MemoryDocument, str]] = []
    for document in documents:
        partitions.extend((document, content) for _, content in split_h1(document.content))
    ranks = bm25_rank(query, [content for _, content in partitions], k1=k1, b=b)
    results: list[MemoryDocument] = []
    seen: set[str] = set()
    for index, _ in ranks:
        document, content = partitions[index]
        if document.metadata.id in seen:
            continue
        seen.add(document.metadata.id)
        results.append(document.model_copy(update={"content": content}))
        if len(results) >= limit:
            break
    return results


__all__ = ["bm25_documents", "bm25_partitions", "bm25_rank", "split_h1"]
