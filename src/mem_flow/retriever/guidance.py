"""Dataset-independent, query-local answer guidance."""

from __future__ import annotations

import re


def answer_task_guidance(query: str) -> str:
    """Return a compact operation checklist without domain or answer hints."""

    normalized = " ".join(query.casefold().split())
    guidance = [
        "Use only selected evidence. Match the requested subject, relation, scope, status, "
        "and time range before deriving the concise answer."
    ]

    if re.search(r"\b(?:how many|count|number of)\b", normalized):
        guidance.append(
            "Enumerate distinct matching entities or completed events, deduplicate source "
            "copies by provenance and event identity, then count the enumeration."
        )
        guidance.append(
            "If the question asks for a stated numeric attribute such as copies, capacity, "
            "price, or duration, return the explicit quantity bound to the closest matching "
            "entity instead of counting evidence records."
        )
        guidance.append(
            "Treat explicit completed participation—attending, competing, volunteering, "
            "or completing an activity at the named event—as an in-scope event even when "
            "the source does not repeat the question's exact participation verb."
        )
        guidance.append(
            "Use the explicit completed records as the closed evidence set. Do not abstain "
            "merely because other vague mentions lack dates or details; exclude those vague "
            "mentions and count the fully supported records."
        )
    if re.search(r"\b(?:how much|total|sum|combined|altogether)\b", normalized):
        guidance.append(
            "Collect every in-scope value with compatible units before summing; do not mix "
            "goals, recommendations, fees, or unrelated values with completed user facts."
        )
        guidance.append(
            "Sum the explicit amounts for completed in-scope expenses. An unpriced planned "
            "or merely mentioned item does not make that known-spending total unknowable."
        )
    if re.search(r"\b(?:difference|more|less|higher|lower|save|increase|decrease)\b", normalized):
        guidance.append(
            "Bind each value to its entity and unit, identify the requested direction, and "
            "show the minimal subtraction or comparison."
        )
        guidance.append(
            "A stated qualitative comparison such as twice, triple, or half is itself a "
            "valid relative answer when the question asks how values compare; do not require "
            "an absolute price or amount unless the question asks for one."
        )
    if re.search(
        r"\b(?:need(?:ed)? to|remaining|left to|short of)\b.{0,80}"
        r"\b(?:earn|reach|redeem|goal|target|threshold)\b",
        normalized,
    ):
        guidance.append(
            "Identify the current progress and required threshold, then subtract current "
            "from target. Return the remaining amount, not either endpoint."
        )
    if re.search(r"\b(?:percent|percentage|ratio|proportion)\b", normalized):
        guidance.append(
            "Identify the numerator and denominator for the same scope before calculating; "
            "do not infer a missing component."
        )
    if re.search(r"\b(?:latest|current|now|previous|initial|used to|changed)\b", normalized):
        guidance.append(
            "Order states for the same entity by event time, observation time, and sequence; "
            "select the state requested rather than collapsing the history."
        )
    if re.search(
        r"\b(?:when|before|after|ago|elapsed|passed|earliest|latest|chronological|order)\b",
        normalized,
    ):
        guidance.append(
            "Resolve both temporal endpoints independently, keep event and observation dates "
            "distinct, and answer in the requested unit or order."
        )
        guidance.append(
            "When the question requests a coarse elapsed unit such as weeks, months, or years, "
            "report completed whole units in ordinary language; retain an extra day only as "
            "an optional clarification instead of converting the result to a decimal."
        )
        guidance.append(
            "For an already-completed event with no explicit event date, use the supporting "
            "source session or observation date as the ordering fallback. A later retelling "
            "does not replace the earliest direct report."
        )
        guidance.append(
            "When the requested relation is between named weekdays, explicit weekday "
            "adjacency can establish before/after even if separate retellings resolve the "
            "relative calendar wording from different observation dates."
        )
    if re.search(r"\bhow old\b.{0,60}\b(?:born|birth)\b", normalized):
        guidance.append(
            "When only whole-year ages are recorded and exact birthdays are unavailable, "
            "use the difference between the stated ages as the conventional approximate "
            "age instead of abstaining over a possible one-year boundary."
        )
    if re.search(
        r"\bhow (?:old|many years) (?:will|would) i be\b.*\bwhen\b",
        normalized,
    ):
        guidance.append(
            "Resolve the current age and the time until the named future event as separate "
            "inputs, then add the elapsed whole years to the current age."
        )
    if re.search(r"\b(?:last|past|previous)\s+(?:weekend|weekday|week|month)\b", normalized):
        guidance.append(
            "When the query repeats the source's relative-time wording to identify an event, "
            "do not discard the unique matching event merely because the query and source "
            "were recorded on different dates; apply strict calendar arithmetic only when "
            "the requested answer itself is a date or elapsed duration."
        )
    if re.search(r"\b(?:first|second|third|fourth|fifth|\d+(?:st|nd|rd|th))\b", normalized):
        guidance.append(
            "For an ordinal request, use the preserved order within the relevant source list; "
            "do not combine entries from unrelated lists."
        )
    if re.search(
        r"\b(?:recommend|suggest|advice|personalize|(?:what do|do) you think)\b",
        normalized,
    ):
        guidance.append(
            "Personalize only from relevant recorded preferences, constraints, tools, and "
            "experience; distinguish them from assistant suggestions the user did not adopt."
        )
        guidance.append(
            "Recorded likes, dislikes, and consumption history are sufficient personalization "
            "evidence. Summarize the matching criteria or reuse evidence examples rather than "
            "abstaining only because no earlier recommendation was stored."
        )
        guidance.append(
            "For a new open-ended request, return a grounded preference profile or decision "
            "direction even when no exact current listing or prior answer is stored; do not "
            "invent external names or facts."
        )
    if re.search(r"\b(?:from whom|who (?:gave|sent|provided)|where did|which person)\b", normalized):
        guidance.append(
            "The question's noun phrase identifies the target event. If one evidence event "
            "uniquely matches its action and time, answer the requested person or place even "
            "when the source uses a narrower or broader name for the object."
        )
    if re.search(r"\b(?:what did you|you (?:said|listed|gave|recommended)|remind me)\b", normalized):
        guidance.append(
            "This asks about assistant-provided content: preserve the source list, wording, "
            "and attributes instead of substituting user facts."
        )

    guidance.append(
        "Resolve omitted subjects, objects, and locations from the nearest unambiguous turns "
        "within the same source conversation; do not carry them across unrelated sources."
    )
    guidance.append(
        "Return only the attribute or list requested. Do not append related accessories, "
        "alternatives, or background facts unless they are needed to disambiguate the answer."
    )

    guidance.append(
        "If any required input is missing or unresolved, state that the stored memory is "
        "insufficient instead of guessing."
    )
    return "\n".join(f"- {item}" for item in guidance)


__all__ = ["answer_task_guidance"]
