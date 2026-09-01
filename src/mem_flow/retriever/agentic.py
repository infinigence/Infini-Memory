"""Bounded DeepAgents retrieval with deterministic BM25 fallback."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..config import RetrievalConfig
from ..llm import FlowLLM
from ..models import DocumentSelection, MemoryDocument
from ..utils.codec import strip_yaml_front_matter
from ..utils.parsing import parse_json_model
from ..utils.tokens import estimate_tokens
from .fact_index import build_fact_query_plan, rank_fact_documents
from .ranking import bm25_partitions, bm25_rank


_SYSTEM_PROMPT = """You are a high-recall memory retrieval agent. Select the smallest complete set
of documents that lets a separate answer model solve QUERY. Use catalog summaries first, then
bounded tools to verify evidence. Prefer one targeted BM25 search before grep/read_lines, and stop
once the required evidence is covered.

Retrieval rules:
- User- and assistant-authored memories are both valid evidence when relevant.
- For aggregates or comparisons, retrieve all in-scope facts needed for the operation instead of
  stopping at the first match. Keep repeated descriptions separate from distinct events.
- For temporal or state-change questions, retrieve the relevant endpoints and preserve their
  dates, sequence, and source attribution.
- For personalization questions, retrieve preferences and constraints that are relevant to the
  requested decision.
- Search with semantic content words rather than date tokens alone. Exclude documents that are
  merely topically similar but contain no evidence needed by the question.

After using the tools, finish with exactly one compact JSON object and no Markdown:
{{"document_ids":["exact_memory_id"]}}
Never invent ids, copy ids exactly, and return at most {limit} ids. Prefer direct evidence from
document bodies. An empty selection must be {{"document_ids":[]}}.
"""

_EVIDENCE_TOOL_NAMES = {"grep", "read_lines", "search"}
_HIGH_RECALL_QUERY_RE = re.compile(
    r"\b(?:how many\s+(?:days?|weeks?|months?|years?)|how many|how old|"
    r"how much|percentage|discount|total(?:led)?|altogether|different|"
    r"most|least|higher|lower|more|less|older|younger|difference|compared|"
    r"current|currently|now|previous|previously|before|after|switch|switched|"
    r"order|earliest|latest|most\s+recent(?:ly)?|first\s+to\s+last|"
    r"which\s+.{0,80}\s+first|"
    r"ago|last\s+(?:day|week|month|year)|"
    r"highest|lowest|maximum|minimum|"
    r"largest|smallest)\b",
    re.IGNORECASE,
)
_PROGRESS_PAIR_QUERY_RE = re.compile(
    r"\b(?:need(?:ed)?\s+to|remaining|left\s+to|short\s+of)\b.{0,80}"
    r"\b(?:earn|reach|redeem|goal|target|threshold)\b",
    re.IGNORECASE,
)
_PERSONALIZATION_QUERY_RE = re.compile(
    r"\b(?:recommend|recommendation|suggest|suggestion|personaliz|"
    r"tips?|advice|ideas?|help(?:ful)?|"
    r"(?:what do|do) you think|"
    r"what (?:should|would)|which .{0,30}(?:should|would)|looking for)\b",
    re.IGNORECASE,
)
_DIRECT_EVIDENCE_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "brand",
    "current",
    "currently",
    "did",
    "do",
    "for",
    "from",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "last",
    "long",
    "me",
    "my",
    "of",
    "on",
    "the",
    "type",
    "to",
    "was",
    "weekend",
    "what",
    "when",
    "where",
    "which",
    "who",
}
_FOCUS_QUERY_EXPANSIONS = {
    "buy": ("bought", "purchase", "purchased", "acquire", "acquired", "ordered"),
    "bought": ("buy", "purchase", "purchased", "acquire", "acquired", "ordered"),
    "purchase": ("buy", "bought", "purchased", "acquire", "acquired", "ordered"),
    "acquire": (
        "acquired",
        "buy",
        "bought",
        "purchase",
        "purchased",
        "received",
        "obtained",
    ),
    "acquired": (
        "acquire",
        "buy",
        "bought",
        "purchase",
        "purchased",
        "received",
        "obtained",
    ),
    "complete": ("completed", "finish", "finished", "done"),
    "completed": ("complete", "finish", "finished", "done"),
    "finish": ("finished", "complete", "completed", "read", "done"),
    "finished": ("finish", "complete", "completed", "read", "done"),
    "beat": ("beaten", "finish", "finished", "complete", "completed", "won"),
    "beaten": ("beat", "finish", "finished", "complete", "completed", "won"),
    "attend": ("attended", "participate", "participated", "visited", "went", "joined"),
    "attended": ("attend", "participate", "participated", "visited", "went", "joined"),
    "participate": (
        "participated",
        "compete",
        "competed",
        "completed",
        "joined",
        "played",
        "ran",
        "took part",
    ),
    "participated": (
        "participate",
        "compete",
        "competed",
        "completed",
        "joined",
        "played",
        "ran",
        "took part",
    ),
    "start": ("started", "begin", "began"),
    "started": ("start", "begin", "began"),
    "reach": ("reached", "arrive", "arrived", "got there"),
    "reached": ("reach", "arrive", "arrived", "got there"),
    "change": ("changed", "update", "updated", "previous", "current"),
    "changed": ("change", "update", "updated", "previous", "current"),
    "current": ("currently", "now", "latest", "active"),
    "previous": ("previously", "former", "earlier", "old"),
    "increase": ("increased", "grew", "growth", "higher", "more"),
    "decrease": ("decreased", "reduced", "lower", "less"),
    "cost": ("price", "paid", "spent", "amount", "fee"),
    "price": ("cost", "paid", "spent", "amount", "fee"),
    "spend": ("spent", "pay", "paid", "cost", "amount", "fee"),
    "spent": ("spend", "pay", "paid", "cost", "amount", "fee"),
    "pay": ("paid", "spend", "spent", "cost", "amount", "fee"),
    "paid": ("pay", "spend", "spent", "cost", "amount", "fee"),
    "discount": ("coupon", "promotion", "savings", "percent", "percentage"),
    "cashback": ("cash back", "rebate", "reward", "rewards", "earned", "savings"),
    "submit": ("submitted", "turn in", "turned in", "hand in", "handed in", "filed"),
    "submitted": ("submit", "turn in", "turned in", "hand in", "handed in", "filed"),
    "duration": (
        "long",
        "elapsed",
        "minutes",
        "hours",
        "days",
        "weeks",
        "months",
        "years",
    ),
    "recommend": ("recommendation", "suggest", "suggestion", "advice", "preference"),
    "recommendation": ("recommend", "suggest", "suggestion", "advice", "preference"),
    "own": ("owns", "owned", "has", "bought", "acquired"),
    "replace": ("replaced", "replacement", "upgrade", "upgraded", "successor"),
    "repair": ("repaired", "fix", "fixed", "maintenance"),
    "fix": ("fixed", "repair", "repaired", "restore", "restored", "maintenance"),
    "fixed": ("fix", "repair", "repaired", "restore", "restored", "maintenance"),
    "cancel": ("cancelled", "canceled", "inactive", "ended"),
    "subscribe": ("subscribed", "subscription", "active", "cancelled"),
    "subscription": ("subscribe", "subscribed", "active", "cancelled"),
    "subscriptions": ("subscription", "subscribe", "subscribed", "active", "cancelled"),
    "old": ("age", "years", "current", "earlier"),
    "born": ("birth", "age", "years", "current"),
    "first": ("earliest", "initial", "before"),
    "latest": ("current", "newest", "most recent", "now"),
    "total": ("sum", "combined", "altogether"),
    "count": ("number", "how many"),
    "relative": ("relation", "family", "person"),
    "assistant": ("you said", "you listed", "you recommended", "provided"),
    # Broad category nouns are weak lexical queries by themselves.  These
    # small ontology families let a query such as "sports events" retrieve a
    # source that naturally says "triathlon" or "tournament" without adding
    # benchmark entities, dates, or expected answers.
    "event": (
        "activity",
        "occasion",
        "festival",
        "concert",
        "race",
        "tournament",
        "match",
    ),
    "events": (
        "activities",
        "occasions",
        "festivals",
        "concerts",
        "races",
        "tournaments",
        "matches",
    ),
    "sport": ("athletic", "game", "match", "race", "run", "tournament", "triathlon"),
    "sports": (
        "athletic",
        "games",
        "matches",
        "races",
        "runs",
        "tournaments",
        "triathlons",
    ),
    "trip": ("travel", "journey", "vacation", "camping", "hike", "road trip"),
    "trips": ("travel", "journeys", "vacations", "camping", "hikes", "road trips"),
    "musical": ("music", "concert", "festival", "jazz", "show"),
    "museum": ("gallery", "exhibition"),
    "museums": ("galleries", "exhibitions"),
    "dog": ("canine", "puppy", "pet", "collar", "leash", "breed"),
    "dogs": ("canines", "puppies", "pets", "collars", "leashes", "breeds"),
    "breed": ("type", "dog", "cat", "canine", "feline", "pet"),
    "game": ("video game", "console", "dlc", "expansion", "played", "finished"),
    "games": ("video games", "consoles", "dlc", "expansions", "played", "finished"),
    "streaming": ("listen", "listening", "music", "songs", "platform", "app"),
    "service": ("platform", "provider", "app", "subscription"),
    "services": ("platforms", "providers", "apps", "subscriptions"),
    "airline": ("airlines", "flight", "flights", "flew", "fly", "carrier", "travel"),
    "airlines": ("airline", "flight", "flights", "flew", "fly", "carriers", "travel"),
    "flight": ("flights", "airline", "flew", "fly", "travel"),
    "flights": ("flight", "airlines", "flew", "fly", "travel"),
    "flew": ("fly", "flight", "flights", "airline", "travelled"),
    "clothing": (
        "clothes",
        "garment",
        "shirt",
        "dress",
        "pants",
        "jeans",
        "boots",
        "blazer",
    ),
    "clothes": (
        "clothing",
        "garments",
        "shirts",
        "dresses",
        "pants",
        "jeans",
        "boots",
        "blazers",
    ),
    "plant": ("plants", "flower", "herb", "succulent", "houseplant", "tree"),
    "plants": ("plant", "flowers", "herbs", "succulents", "houseplants", "trees"),
    "furniture": (
        "table",
        "chair",
        "desk",
        "sofa",
        "couch",
        "shelf",
        "bookcase",
        "cabinet",
        "dresser",
        "bed",
        "bedroom",
    ),
    "album": ("albums", "ep", "record", "records", "release", "music"),
    "albums": ("album", "ep", "eps", "records", "releases", "music"),
    "download": ("downloaded", "purchase", "purchased", "bought", "saved"),
    "downloaded": ("download", "purchase", "purchased", "bought", "saved"),
    "bake": ("baked", "baking", "made", "cooked"),
    "baked": ("bake", "baking", "made", "cooked"),
    "delivery": ("deliver", "delivered", "order", "ordered", "takeout", "courier"),
    "raise": ("raised", "fundraise", "fundraising", "collected", "donations"),
    "raised": ("raise", "fundraise", "fundraising", "collected", "donations"),
    "workshop": ("workshops", "class", "training", "seminar", "course"),
    "workshops": ("workshop", "classes", "training", "seminars", "courses"),
    "episode": ("episodes", "podcast", "listened", "show"),
    "episodes": ("episode", "podcasts", "listened", "shows"),
    "property": ("properties", "home", "house", "apartment", "condo", "townhouse"),
    "properties": ("property", "homes", "houses", "apartments", "condos", "townhouses"),
    "antique": ("antiques", "vintage", "heirloom", "collectible", "old"),
    "antiques": ("antique", "vintage", "heirlooms", "collectibles", "old"),
    "inherit": ("inherited", "heirloom", "passed down", "family", "received"),
    "inherited": ("inherit", "heirloom", "passed down", "family", "received"),
    "novel": ("novels", "book", "books", "read"),
    "novels": ("novel", "book", "books", "read"),
    "page": ("pages", "page count", "length"),
    "pages": ("page", "page count", "length"),
    "publication": (
        "publications",
        "paper",
        "article",
        "journal",
        "conference",
        "research",
    ),
    "publications": (
        "publication",
        "papers",
        "articles",
        "journals",
        "conferences",
        "research",
    ),
    "conference": (
        "conferences",
        "publication",
        "paper",
        "journal",
        "research",
        "symposium",
    ),
    "battery": ("power", "charge", "charging", "charger", "power bank", "phone"),
    "phone": ("mobile", "smartphone", "battery", "charger", "power bank", "charging"),
    "cocktail": ("cocktails", "drink", "drinks", "mixology", "recipe", "bartending"),
    "reunion": (
        "school",
        "classmate",
        "friends",
        "graduation",
        "high school",
        "alumni",
    ),
    "creamer": (
        "coffee",
        "recipe",
        "milk",
        "vanilla",
        "sweetener",
        "sugar",
        "homemade",
    ),
    "movie": ("movies", "film", "films", "cinema", "watched", "viewing"),
    "movies": ("movie", "film", "films", "cinema", "watched", "viewing"),
    "film": ("films", "movie", "movies", "cinema", "watched", "viewing"),
    "doctor": (
        "doctors",
        "physician",
        "specialist",
        "clinic",
        "medical",
        "appointment",
    ),
    "doctors": (
        "doctor",
        "physicians",
        "specialists",
        "clinic",
        "medical",
        "appointments",
    ),
    "gpa": (
        "grade",
        "grades",
        "academic",
        "undergraduate",
        "graduate",
        "college",
        "university",
    ),
    "homegrown": (
        "garden",
        "produce",
        "vegetable",
        "fruit",
        "herb",
        "ingredients",
        "harvest",
    ),
    "graduate": (
        "graduated",
        "graduation",
        "degree",
        "college",
        "university",
        "school",
    ),
    "graduated": (
        "graduate",
        "graduation",
        "degree",
        "college",
        "university",
        "school",
    ),
    "theme": ("amusement", "attraction", "ride", "rides", "park", "festival", "show"),
    "park": ("parks", "amusement", "attraction", "ride", "rides", "festival", "show"),
    "bike": ("bicycle", "cycling", "ride", "chain", "cassette", "wheel", "computer"),
    "bikes": ("bicycles", "cycling", "rides", "chains", "cassettes", "wheels"),
    "sibling": ("siblings", "brother", "sister", "family"),
    "siblings": ("sibling", "brothers", "sisters", "family"),
    "charity": (
        "fundraiser",
        "fundraising",
        "donation",
        "donations",
        "raised",
        "benefit",
    ),
    "clinic": ("doctor", "appointment", "hospital", "medical", "arrived", "reached"),
    "gadget": ("appliance", "device", "equipment", "tool", "kitchen"),
    "appliance": ("gadget", "device", "equipment", "cooker", "oven", "kitchen"),
}


class AgentSelectionError(ValueError):
    """Raised when an agent result contains neither a selection nor tool evidence."""


_CATALOG_QUERY_EXACT = {
    "allmemories",
    "allmemory",
    "everythingyouremember",
    "listmemories",
    "listmymemories",
    "showmemories",
    "showmymemories",
    "whatdoyouremember",
    "你的记忆",
    "你记得什么",
    "你记得哪些内容",
    "你记住了什么",
    "列出你的记忆",
    "列出所有记忆",
    "列出记忆内容",
    "我的记忆",
    "所有记忆内容",
    "查看记忆内容",
    "检索你的记忆内容",
    "检索记忆内容",
    "记忆内容",
}


def _is_catalog_query(query: str) -> bool:
    """Return whether *query* asks to browse memory rather than match a fact."""

    normalized = re.sub(r"[\W_]+", "", query.casefold())
    if normalized in _CATALOG_QUERY_EXACT:
        return True
    if "记忆" in normalized and any(
        marker in normalized
        for marker in ("所有", "全部", "全量", "完整", "一切", "有哪些")
    ):
        return True
    mentions_memory = "memory" in normalized or "memories" in normalized
    return mentions_memory and (
        normalized.startswith("all") or "everything" in normalized
    )


def _needs_high_recall_merge(query: str) -> bool:
    """Return whether a query is likely to aggregate evidence across documents."""

    return bool(_HIGH_RECALL_QUERY_RE.search(query))


def _needs_progress_pair(query: str) -> bool:
    """Return whether the answer needs one current value and one target value."""

    return bool(_PROGRESS_PAIR_QUERY_RE.search(query))


def _needs_personalization_context(query: str) -> bool:
    """Return whether an answer should retain complete preference context."""

    return bool(_PERSONALIZATION_QUERY_RE.search(query))


def _generic_lexical_variants(token: str) -> tuple[str, ...]:
    """Return conservative English inflections without domain vocabulary."""

    variants: set[str] = set()
    if token.endswith("ies") and len(token) > 4:
        variants.add(token[:-3] + "y")
    elif token.endswith(("sses", "shes", "ches", "xes", "zes", "oes")):
        variants.add(token[:-2])
    elif token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        variants.add(token[:-1])
    if token.endswith("ied") and len(token) > 4:
        variants.add(token[:-3] + "y")
    elif token.endswith("ed") and len(token) > 4:
        variants.update({token[:-2], token[:-1]})
    if token.endswith("ing") and len(token) > 5:
        variants.update({token[:-3], token[:-3] + "e"})
    return tuple(sorted(variant for variant in variants if len(variant) >= 3))


def _query_token_expansions(token: str) -> tuple[str, ...]:
    """Return generic inflections and semantic variants for one query token."""

    inflections = _generic_lexical_variants(token)
    variants = set(_FOCUS_QUERY_EXPANSIONS.get(token, ()))
    variants.update(inflections)
    for inflection in inflections:
        variants.update(_FOCUS_QUERY_EXPANSIONS.get(inflection, ()))
    variants.discard(token)
    return tuple(sorted(variants))


def _expand_focus_query(query: str) -> str:
    """Add narrow lexical variants used by fact-line BM25 focusing."""

    # Evaluation and API callers may prepend a reference timestamp.  Date
    # tokens are useful to the answer model but make poor lexical retrieval
    # terms: they can outscore the entity and action in the actual question.
    query_body = query.rsplit("Question:", 1)[-1].strip()
    lowered = {token.lower() for token in re.findall(r"[A-Za-z]+", query_body)}
    service_is_category = bool(
        re.search(
            r"\b(?:streaming|delivery|subscription|online|cloud)\s+services?\b",
            query_body,
            re.I,
        )
    )
    expansions = [
        variant
        for token in lowered
        if not (token in {"service", "services"} and service_is_category)
        for variant in _query_token_expansions(token)
    ]
    return " ".join([query_body, *expansions])


def _has_direct_lexical_evidence(query: str, documents: list[MemoryDocument]) -> bool:
    """Detect a rare query concept directly present in top BM25 leaves.

    Direct factual questions often need one answer call, not a multi-turn tool
    loop.  Semantic/no-overlap queries still fall through to the LLM agent.
    """

    query_body = query.rsplit("Question:", 1)[-1].strip()
    concepts = {
        token
        for token in re.findall(r"[a-z0-9]+", query_body.casefold())
        if len(token) >= 4 and token not in _DIRECT_EVIDENCE_STOPWORDS
    }
    if not concepts:
        return False
    for document in documents[:3]:
        body = strip_yaml_front_matter(document.content).casefold()
        matched = {
            concept
            for concept in concepts
            if re.search(rf"\b{re.escape(concept)}\b", body)
            or any(
                re.search(rf"\b{re.escape(variant.casefold())}\b", body)
                for variant in _query_token_expansions(concept)
            )
        }
        required = 1 if len(concepts) <= 2 else 2
        if len(matched) >= required:
            return True
    return False


def _focus_subqueries(query: str) -> list[str]:
    """Build entity-balanced searches for aggregation questions.

    A single BM25 query can be dominated by the most frequently discussed
    entity (for example many facts about one subgoal hiding another). Each
    known semantic family therefore gets a small independent retrieval budget.
    The full query remains the final fallback, so this only improves recall and
    does not replace the agent's semantic selection.
    """

    lowered = {token.lower() for token in re.findall(r"[A-Za-z]+", query)}
    return [
        " ".join((token, *_FOCUS_QUERY_EXPANSIONS[token]))
        for token in sorted(lowered)
        if token in _FOCUS_QUERY_EXPANSIONS
    ]


class AgenticRetriever(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    documents: list[MemoryDocument]
    llm: FlowLLM
    config: RetrievalConfig
    limit: int

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger("mem_flow.retriever.agentic")

    @property
    def by_id(self) -> dict[str, MemoryDocument]:
        return {item.metadata.id: item for item in self.documents}

    def _fallback(self, query: str) -> list[MemoryDocument]:
        expanded_query = _expand_focus_query(query)
        structured_limit = max(1, (self.limit + 1) // 2)
        structured = rank_fact_documents(
            query,
            self.documents,
            structured_limit,
            expanded_query=expanded_query,
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
        )
        lexical = bm25_partitions(
            expanded_query,
            self.documents,
            self.limit,
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
        )
        broad_hits = _merge_documents(structured, lexical, self.limit)
        if _needs_high_recall_merge(query) or _needs_personalization_context(query):
            # Counts, totals, timelines, and personalization require facts
            # spread across broad consolidated leaves.  Reserving half the
            # result for individual source sessions displaced complementary
            # events and caused a large multi-session accuracy regression.
            return broad_hits
        # Eval lossless ingestion and production callers may retain atomic
        # source-conversation evidence beside broader consolidated leaves.
        # Reserve recall slots for those session-sized records: otherwise a
        # large memory leaf containing dozens of sessions wins many lexical
        # terms while hiding the one turn needed for an exact answer.
        evidence = [
            document
            for document in self.documents
            if document.metadata.kind == "evidence"
        ]
        evidence_limit = min(8, max(2, self.limit // 2))
        evidence_hits = bm25_partitions(
            expanded_query,
            evidence,
            evidence_limit,
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
        )
        if build_fact_query_plan(query).assistant_source:
            return _merge_documents(evidence_hits, broad_hits, self.limit)
        # Consolidated atomic bullets preserve normalized user facts and are
        # much less noisy than full source conversations.  Keep fact-index hits
        # first, then exact source evidence, then lexical-only memory leaves.
        # This prevents a generic topic document from hiding an exact source
        # turn while still preferring a directly matching normalized fact over
        # incidental terms in a long conversation.  Assistant-authored recall
        # deliberately reverses the order above because exact wording matters.
        return _merge_documents(
            structured,
            _merge_documents(evidence_hits, lexical, self.limit),
            self.limit,
        )

    def _balanced_fallback(self, query: str) -> list[MemoryDocument]:
        """Reserve recall slots for each query entity before whole-query BM25."""

        result = rank_fact_documents(
            query,
            self.documents,
            max(1, (self.limit + 1) // 2),
            expanded_query=_expand_focus_query(query),
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
        )
        for subquery in _focus_subqueries(query):
            result = _merge_documents(
                result,
                bm25_partitions(
                    subquery,
                    self.documents,
                    min(2, self.limit),
                    k1=self.config.bm25_k1,
                    b=self.config.bm25_b,
                ),
                self.limit,
            )
        return _merge_documents(result, self._fallback(query), self.limit)

    @staticmethod
    def _focus_document_for_answer(
        query: str,
        document: MemoryDocument,
        *,
        max_lines: int,
    ) -> MemoryDocument:
        """Return the highest-scoring fact lines from one selected document."""

        heading = ""
        role = ""
        candidates: list[str] = []
        for raw_line in strip_yaml_front_matter(document.content).splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("# "):
                heading = line
                continue
            if line in {"[USER]", "[ASSISTANT]", "[SYSTEM]", "[TOOL]"}:
                role = line
                continue
            prefix = "\n".join(item for item in (heading, role) if item)
            candidates.append(f"{prefix}\n{line}" if prefix else line)
        ranks = bm25_rank(
            _expand_focus_query(query),
            candidates,
            k1=1.5,
            b=0.75,
        )
        if not ranks:
            return document
        # An isolated top-scoring line often contains the user's question or
        # action while the answer-bearing entity is in the adjacent assistant
        # response, list item, or clarification.  Expand a few strong seeds to
        # a bounded local window, then fill any remaining slots by global rank.
        # Preserve source order for timelines, old/new states, and ordinals.
        seed_count = min(max_lines, max(2, max_lines // 3))
        seeds = [index for index, _score in ranks[:seed_count]]
        selected_set: set[int] = set(seeds)
        for radius in (1, 2):
            for index in seeds:
                for nearby in (index - radius, index + radius):
                    if len(selected_set) >= max_lines:
                        break
                    if 0 <= nearby < len(candidates):
                        selected_set.add(nearby)
        for index, _score in ranks:
            if len(selected_set) >= max_lines:
                break
            selected_set.add(index)
        selected_indices = sorted(selected_set)
        focused = "\n".join(candidates[index] for index in selected_indices)
        return document.model_copy(update={"content": focused})

    def _focus_documents_for_answer(
        self, query: str, documents: list[MemoryDocument]
    ) -> list[MemoryDocument]:
        # BM25 partition retrieval returns a matching H1 fragment while keeping
        # the leaf document id. Rehydrate those hits before the answer pass so
        # another section in the same leaf is not silently lost when hits are
        # deduplicated by id.
        complete_documents = [
            self.by_id.get(document.metadata.id, document) for document in documents
        ]
        wide_context = _needs_high_recall_merge(
            query
        ) or _needs_personalization_context(query)
        max_lines = (
            self.config.agentic_aggregation_max_lines_per_document
            if wide_context
            else self.config.agentic_answer_max_lines_per_document
        )
        assistant_source = build_fact_query_plan(query).assistant_source or bool(
            re.search(
                r"\b(?:you (?:listed|provided|gave|said|recommended)|"
                r"previous (?:chat|conversation)|remind me what you)\b",
                query,
                re.I,
            )
        )
        return [
            # Evidence documents are already session-sized authoritative
            # sources.  Per-document focusing can keep the matching user
            # request while silently dropping a later list item, budget line,
            # or final bullet from the assistant response.  Preserve evidence
            # for explicit assistant-memory queries and let the downstream
            # bounded cross-document ledger perform the only line-ranking pass.
            # All other documents are bounded, including high-recall queries:
            # sending twenty complete long-term leaves made answer calls exceed
            # the service timeout even though only a few lines in each leaf
            # were relevant.
            document
            if document.metadata.kind == "evidence" and assistant_source
            else self._focus_document_for_answer(
                query,
                document,
                max_lines=max_lines,
            )
            for document in complete_documents
        ]

    def _grep(self, pattern: str, limit: int | None = None) -> dict[str, object]:
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return {"error": "invalid regex", "matches": []}
        resolved_limit = min(
            max(1, limit or self.config.agentic_grep_limit),
            self.config.agentic_grep_limit,
        )
        matches: list[dict[str, object]] = []
        for document in self.documents:
            for number, line in enumerate(document.content.splitlines(), start=1):
                if compiled.search(line):
                    matches.append(
                        {
                            "document_id": document.metadata.id,
                            "line_number": number,
                            "line": line[:500],
                        }
                    )
                    if len(matches) >= resolved_limit:
                        return {"matches": matches, "truncated": True}
        return {"matches": matches, "truncated": False}

    def _search(self, query: str, limit: int | None = None) -> list[dict[str, object]]:
        resolved_limit = min(max(1, limit or self.limit), self.limit)
        documents = bm25_partitions(
            query,
            self.documents,
            resolved_limit,
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
        )
        return [
            {
                "document_id": item.metadata.id,
                "title": item.metadata.title,
                "snippet": item.content[:1000],
            }
            for item in documents
        ]

    def _list_docs(
        self, offset: int = 0, limit: int | None = None
    ) -> dict[str, object]:
        offset = max(0, offset)
        page_size = min(
            max(1, limit or self.config.agentic_list_docs_page_size),
            self.config.agentic_list_docs_page_size,
        )
        page = self.documents[offset : offset + page_size]
        summary_limit = self.config.agentic_catalog_summary_length
        return {
            "total": len(self.documents),
            "documents": [
                {
                    "id": item.metadata.id,
                    "title": item.metadata.title,
                    "summary": item.metadata.summary[:summary_limit],
                }
                for item in page
            ],
            "has_more": offset + page_size < len(self.documents),
        }

    def _read_lines(
        self, document_id: str, start_line: int = 1, end_line: int | None = None
    ) -> dict[str, object]:
        document = self.by_id.get(document_id)
        if document is None:
            return {"error": "document not found"}
        lines = document.content.splitlines()
        if not lines:
            return {"document_id": document_id, "lines": ""}
        start = min(max(1, start_line), len(lines))
        requested_end = end_line if end_line is not None else start + 49
        end = min(
            max(start, requested_end),
            start + self.config.agentic_read_lines_max_range - 1,
            len(lines),
        )
        return {
            "document_id": document_id,
            "start_line": start,
            "end_line": end,
            "lines": "\n".join(
                f"{number}: {line}"
                for number, line in enumerate(lines[start - 1 : end], start=start)
            ),
        }

    @staticmethod
    def _final_content(result: Any) -> str:
        messages = result.get("messages", []) if isinstance(result, dict) else []
        for message in reversed(messages):
            if isinstance(message, dict):
                role = message.get("role") or message.get("type")
                if role not in {None, "ai", "assistant"}:
                    continue
                content = message.get("content", "")
            else:
                if message.__class__.__name__ != "AIMessage":
                    continue
                content = getattr(message, "content", "")
            if isinstance(content, list):
                text = "".join(
                    str(block.get("text", ""))
                    for block in content
                    if isinstance(block, dict)
                )
            else:
                text = str(content)
            if text.strip():
                return text
        return ""

    @staticmethod
    def _tool_evidence_document_ids(result: Any) -> list[str]:
        messages = result.get("messages", []) if isinstance(result, dict) else []
        # A document the agent explicitly read is stronger evidence than a
        # broad BM25 result returned earlier in the tool loop.  This ordering
        # is especially important when the graph reaches its recursion bound:
        # the verified evidence should not be buried behind every candidate
        # from the first search call.
        ids_by_tool: dict[str, list[str]] = {
            "read_lines": [],
            "grep": [],
            "search": [],
        }
        for message in messages:
            if isinstance(message, dict):
                name = message.get("name")
                content = message.get("content", "")
            else:
                name = getattr(message, "name", None)
                content = getattr(message, "content", "")
            if name not in _EVIDENCE_TOOL_NAMES or not isinstance(content, str):
                continue
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                continue
            candidates: list[object] = []
            if name == "read_lines" and isinstance(payload, dict):
                candidates = [payload.get("document_id")]
            elif name == "search" and isinstance(payload, list):
                candidates = [
                    item.get("document_id")
                    for item in payload
                    if isinstance(item, dict)
                ]
            elif name == "grep" and isinstance(payload, dict):
                matches = payload.get("matches", [])
                if isinstance(matches, list):
                    candidates = [
                        item.get("document_id")
                        for item in matches
                        if isinstance(item, dict)
                    ]
            for document_id in candidates:
                selected = ids_by_tool[name]
                if isinstance(document_id, str) and document_id not in selected:
                    selected.append(document_id)
        document_ids: list[str] = []
        for tool_name in ("read_lines", "grep", "search"):
            for document_id in ids_by_tool[tool_name]:
                if document_id not in document_ids:
                    document_ids.append(document_id)
        return document_ids

    @staticmethod
    def _stream_agent_result(
        agent: Any,
        payload: dict[str, object],
        *,
        recursion_limit: int,
    ) -> tuple[dict[str, Any], bool]:
        """Keep the latest graph state when a bounded agent exhausts its steps."""

        from langgraph.errors import GraphRecursionError

        latest: dict[str, Any] = {}
        try:
            for state in agent.stream(
                payload,
                config={"recursion_limit": recursion_limit},
                stream_mode="values",
            ):
                if isinstance(state, dict):
                    latest = state
        except GraphRecursionError:
            if not latest:
                raise
            return latest, True
        return latest, False

    @classmethod
    def _selection_from_result(cls, result: Any) -> tuple[DocumentSelection, str]:
        parse_error: Exception | None = None
        if isinstance(result, dict) and result.get("structured_response") is not None:
            try:
                return (
                    DocumentSelection.model_validate(result["structured_response"]),
                    "structured_response",
                )
            except Exception as exc:
                parse_error = exc

        content = cls._final_content(result)
        if content.strip():
            try:
                return parse_json_model(content, DocumentSelection), "final_content"
            except Exception as exc:
                parse_error = exc

        evidence_ids = cls._tool_evidence_document_ids(result)
        if evidence_ids:
            return DocumentSelection(document_ids=evidence_ids), "tool_evidence"

        reason = type(parse_error).__name__ if parse_error else "empty_final_content"
        raise AgentSelectionError(f"agent selection unavailable: {reason}")

    def search(self, query: str) -> list[MemoryDocument]:
        if _is_catalog_query(query):
            selected = self.documents[: self.limit]
            self.logger.info(
                "agentic_catalog_browse candidate_count=%d selected_count=%d limit=%d",
                len(self.documents),
                len(selected),
                self.limit,
            )
            return selected

        if _needs_high_recall_merge(query) or _needs_personalization_context(query):
            # The agent used to consume several slow LLM calls and was then
            # followed by this same deterministic recall merge.  For counts,
            # totals, comparisons, and temporal pairs, go directly to the
            # entity-balanced candidates and let the bounded evidence ledger
            # verify all facts in one answer call.
            if _needs_progress_pair(query):
                selected = self._fallback(query)[: min(self.limit, 4)]
                mode = "progress_pair"
            elif _needs_personalization_context(query):
                selected = self._balanced_fallback(query)[: self.limit]
                mode = "personalization"
            else:
                selected = self._balanced_fallback(query)[: self.limit]
                mode = "high_recall"
            self.logger.info(
                "agentic_deterministic_recall mode=%s candidate_count=%d "
                "selected_count=%d limit=%d",
                mode,
                len(self.documents),
                len(selected),
                self.limit,
            )
            if selected:
                return self._focus_documents_for_answer(query, selected)
            self.logger.info(
                "agentic_deterministic_recall_empty fallback=llm_agent mode=%s",
                mode,
            )

        lexical_candidates = self._fallback(query)[: min(self.limit, 8)]
        if _has_direct_lexical_evidence(query, lexical_candidates):
            self.logger.info(
                "agentic_deterministic_recall mode=direct_lexical "
                "candidate_count=%d selected_count=%d limit=%d",
                len(self.documents),
                len(lexical_candidates),
                self.limit,
            )
            return self._focus_documents_for_answer(query, lexical_candidates)

        model_factory = getattr(self.llm, "as_langchain_model", None)
        if not callable(model_factory):
            self.logger.warning("agentic_model_unavailable fallback=BM25_partition")
            return self._fallback(query)
        try:
            from deepagents import create_deep_agent
            from langchain_core.tools import tool

            @tool
            def grep(pattern: str, limit: int | None = None) -> str:
                """Regex-search all memory document lines."""
                return json.dumps(self._grep(pattern, limit), ensure_ascii=False)

            @tool
            def search(search_query: str, limit: int | None = None) -> str:
                """BM25-search H1 sections and return relevant snippets."""
                return json.dumps(self._search(search_query, limit), ensure_ascii=False)

            @tool
            def list_docs(offset: int = 0, limit: int | None = None) -> str:
                """List a bounded page of document ids and summaries."""
                return json.dumps(self._list_docs(offset, limit), ensure_ascii=False)

            @tool
            def read_lines(
                document_id: str,
                start_line: int = 1,
                end_line: int | None = None,
            ) -> str:
                """Read a bounded line range from one exact document."""
                return json.dumps(
                    self._read_lines(document_id, start_line, end_line),
                    ensure_ascii=False,
                )

            agent = create_deep_agent(
                model=model_factory(),
                tools=[grep, search, list_docs, read_lines],
                system_prompt=_SYSTEM_PROMPT.format(limit=self.limit),
            )
            catalog = [
                {
                    "id": item.metadata.id,
                    "title": item.metadata.title,
                    "summary": item.metadata.summary[
                        : self.config.agentic_catalog_summary_length
                    ],
                }
                for item in self.documents
            ]
            result, recursion_exhausted = self._stream_agent_result(
                agent,
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": json.dumps(
                                {"query": query, "document_catalog": catalog},
                                ensure_ascii=False,
                            ),
                        }
                    ]
                },
                recursion_limit=self.config.agentic_max_iterations * 4,
            )
            selected, selection_source = self._selection_from_result(result)
            hits = [
                self.by_id[document_id]
                for document_id in selected.document_ids
                if document_id in self.by_id
            ][: self.limit]
            self.logger.info(
                "agentic_completed selected_count=%d selection_source=%s "
                "recursion_exhausted=%s",
                len(hits),
                selection_source,
                recursion_exhausted,
            )
        except AgentSelectionError as exc:
            self.logger.warning(
                "agentic_selection_unavailable reason=%s fallback=BM25_partition",
                exc,
            )
            hits = []
        except Exception:
            self.logger.exception("agentic_failed fallback=BM25_partition")
            hits = []
        enough_documents = len(hits) >= self.config.agentic_min_relevant_docs
        enough_content = (
            sum(estimate_tokens(item.content) for item in hits)
            >= self.config.agentic_min_result_tokens
        )
        if _needs_high_recall_merge(query):
            if _needs_progress_pair(query):
                # Goal-progress questions need exactly two kinds of evidence:
                # the current balance and threshold. Broad tool recovery can
                # otherwise add many product/program mentions and make the
                # answer model repeat the threshold instead of subtracting.
                focused_limit = min(self.limit, 4)
                combined = _merge_documents(
                    self._fallback(query)[:focused_limit],
                    hits,
                    focused_limit,
                )
                return self._focus_documents_for_answer(query, combined)
            # An agent can correctly find one relevant document and still stop before
            # discovering the remaining events needed for a count.  Add only the top
            # deterministic candidates so aggregation gets higher recall without
            # flooding the answer model with the full catalog.
            fallback_limit = min(self.limit, 8)
            combined = _merge_documents(
                self._balanced_fallback(query)[:fallback_limit],
                hits,
                self.limit,
            )
            return self._focus_documents_for_answer(query, combined)
        if enough_documents and enough_content:
            return self._focus_documents_for_answer(query, hits)
        combined = _merge_documents(hits, self._fallback(query), self.limit)
        return self._focus_documents_for_answer(query, combined)


def _merge_documents(
    first: list[MemoryDocument], second: list[MemoryDocument], limit: int
) -> list[MemoryDocument]:
    result: list[MemoryDocument] = []
    seen: set[str] = set()
    for document in [*first, *second]:
        if document.metadata.id not in seen:
            seen.add(document.metadata.id)
            result.append(document)
        if len(result) >= limit:
            break
    return result


__all__ = ["AgenticRetriever"]
