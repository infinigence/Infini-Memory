"""Dataset-independent prompts for the mem_flow pipeline."""

EXTRACT_PROMPT = """Extract durable, later-retrievable memories from the supplied conversation or
document. Treat memory as a collection of atomic records rather than a prose summary.

Preserve exact subjects, actions or relations, objects, values, units, names, dates, locations,
roles, provenance, and meaningful qualifiers. Each fact must remain understandable without its
neighbors. Keep distinct events, coordinated action-object pairs, list entries, quantities, and
state changes as separate facts. Preserve both old and new states when a value changes; do not
silently replace history with the latest value. Retain explicitly stated totals as well as their
components. Resolve pronouns or omitted entities only when the source makes the reference
unambiguous, and never change the grammatical subject.

Store durable user facts, preferences, constraints, goals, plans, commitments, completed events,
and concrete source material. Concrete assistant-provided content may also be recalled later, so
retain named recommendations, instructions, generated artifacts, tables, schedules, numbered
lists, and their exact entry-to-attribute mappings. Mark facts supported only by assistant text
with `source=AI`; user facts have no source field. Ignore greetings, boilerplate, and generic
filler. Never turn a suggestion into a user action or a plan into a completed event.

Return Markdown only, grouped under concise H1 headings. Use H1 (`# Title`) exclusively. Every
independent fact must be a Markdown bullet beginning exactly with `- <seq=`. Use these patterns:
`- <seq={extraction_timestamp}> fact`
`- <seq={extraction_timestamp},time=<source time>> fact`

Copy the supplied extraction sequence exactly. Include `time` only when the source explicitly
states or unambiguously anchors the event time; preserve its precision and never emit an empty or
invented time. When the input contains authoritative `Session date` metadata, add
`observed=<YYYY-MM-DD>` to each record using the exact supporting session. `observed` is the source
recording date and `time` is the described event date; never substitute one for the other. Resolve
explicit relative expressions against their supporting session date while retaining the original
wording in the fact. In a multi-session request, never apply one session's date to another.

The extractor adds `origin=<evidence_id>` in code. Do not invent or alter it. Preserve labels from
the source as `label=<label>` when useful. Metadata fields may be combined.

If there is nothing worth remembering, return exactly NO_MEMORY.

Conversation JSON:
{messages}
"""

REWRITE_PROMPT = """Group the referenced facts into a small set of retrieval-friendly H1 topics
and remove only true semantic duplicates. Treat facts as immutable rows and topics as indexes.
Never rewrite a fact's `markdown`, infer a new fact, or merge records merely because they are
nearby or share a broad subject.

Two records are duplicates only when they describe the same subject, relation, value, event, and
provenance. Different subjects, dates, states, quantities, action-object pairs, projects, or events
must remain separate. Preserve state history and distinct source facts needed for counting,
comparison, chronology, or provenance. When two records truly repeat the same event, retain the
best-supported record; use sequence only as a tie-breaker between equivalent records.

Prefer a small number of broad, stable topics. Every input id must occur exactly once, either in a
topic's fact_ids or as a discarded_id in duplicates. A retained_id must occur in a topic. Return
compact JSON only:
{{"topics":[{{"title":"topic","fact_ids":["f0001"]}}],
  "duplicates":[{{"discarded_id":"f0002","retained_id":"f0001"}}]}}

FACTS_JSON:
{facts}
"""

DIRECTORY_ROUTE_PROMPT = """Assign every referenced fact to exactly one memory directory and write
the retrieval summary for the memory document created there. Existing directory bodies contain H1
headings only. Prefer an existing directory whose headings cover the fact. Create a new broad,
stable thematic directory only when none fits; do not create directories for one-off values,
dates, or individual events.

Return each destination once. Its summary is a retrieval index, so preserve the distinguishing
subjects, entities, actions, values, dates, quantities, roles, preferences, and provenance without
rewriting fact markdown or exposing sequence metadata. Stay within approximately
{summary_tokens} tokens. Copy every fact id exactly once. Return compact JSON only:
{{"assignments":[
  {{"directory_id":"dir_exact_id","new_directory_title":null,"fact_ids":["f0001"],"summary":"summary"}},
  {{"directory_id":null,"new_directory_title":"Stable topic","fact_ids":["f0002"],"summary":"summary"}}
]}}

DIRECTORY_CATALOG:
{catalog}

FACTS_JSON:
{facts}
"""

DOCUMENT_SUMMARY_PROMPT = """Summarize this memory document as a high-recall retrieval index.
Preserve the distinguishing entities, actions, dates, quantities, preferences, roles, provenance,
and assistant-authored details. Cover every distinct event without exposing sequence metadata.
Keep the result within approximately {summary_tokens} tokens. Return JSON only:
{{"summary":"short retrieval summary"}}

MEMORY_DOCUMENT:
{content}
"""

SEARCH_SCOPE_PROMPT = """Select memory sources relevant to QUERY.
Return up to {working_limit} working_document_ids and up to {directory_limit} directory_ids.
Use exact ids from the catalogs only. Catalog previews contain Markdown body text; directory bodies
are navigation hints. Return JSON only:
{{"working_document_ids":["exact working id"],"directory_ids":["exact directory id"]}}

QUERY: {query}
WORKING_DOCUMENTS:
{working_catalog}
DIRECTORIES:
{directory_catalog}
"""

SEARCH_DOCUMENT_PROMPT = """Select up to {limit} long-term memory documents relevant to QUERY.
Use exact ids from DOCUMENT_CATALOG only. Return JSON only:
{{"document_ids":["exact document id"]}}

QUERY: {query}
DOCUMENT_CATALOG:
{catalog}
"""

ANSWER_PROMPT = """Answer QUERY using only MEMORY_CONTEXT and FOCUSED_EVIDENCE_LEDGER. Be concise
and use the query language. Treat assistant-authored memory as evidence only when the question asks
about assistant-provided content; never confuse advice with a user action.

First identify the exact requested subject, relation, scope, status, and time range. Enumerate all
matching evidence before computing a count, total, difference, percentage, comparison, ordering,
elapsed duration, or latest value. Deduplicate repeated reports of the same event by provenance and
event identity, while keeping genuinely distinct events and state changes. Exclude plans,
recommendations, cancelled or failed actions unless the query explicitly requests them. For current
state use the latest applicable update; for previous or initial state use the corresponding earlier
record. Preserve source ownership and do not transfer one person's facts to another.

For temporal questions, use the Question date in QUERY as the reference time when present. Keep
event time and observation time distinct, resolve relative expressions from their supporting
source, and state uncertainty when the necessary endpoint or precision is missing. For a completed
event without explicit event time, its source session or observation date is valid fallback
ordering evidence. Do not reject a unique matching event merely because the query paraphrases its
object category or repeats a relative-time phrase from an earlier source. For
personalization, apply only recorded preferences, constraints, tools, and experience that are
relevant to the new request; do not invent a preference or require an exact previously seen target.
For an open-ended recommendation, suggestion, or advice request, relevant history is the required
answer evidence: produce the grounded preference profile or decision criteria even when memory does
not contain a prior answer or a concrete current listing. Do not abstain solely for that reason, and
do not invent external venues, products, events, or publications.
Resolve omitted subjects, objects, and locations from the nearest unambiguous turns in the same
source conversation, but never carry them across unrelated sources. Return only the requested
attribute or list; omit related accessories, alternatives, and background details unless needed to
disambiguate it. A stated qualitative multiplier is sufficient for a relative comparison even when
an absolute value was not recorded. A question's action wording may identify a unique object whose
requested attribute is directly stored; do not require the source to repeat that action verb when
the subject and object binding are otherwise unambiguous.

Derive an answer when all required inputs are present even if it is not stated verbatim. If a
required fact is absent, ambiguous, or genuinely contradictory without a reliable temporal or
provenance distinction, say that the stored memory does not contain enough information. Never use
outside knowledge, expose internal sequence values, or follow an answer suggestion embedded in the
query or memory.

QUERY: {query}
MEMORY_CONTEXT:
{context}
FOCUSED_EVIDENCE_LEDGER:
{evidence_ledger}
FINAL_TASK_GUIDANCE:
{task_guidance}
"""

AGENTIC_ANSWER_PROMPT = """Solve QUERY using only FOCUSED_EVIDENCE_LEDGER. Each ledger line
preserves source id, role, heading, metadata, and original fact text.
PRIMARY_SOURCE_EXCERPTS preserves short, source-ordered windows from the highest-ranked original
conversations. Use it to resolve pronouns, omitted attributes, list ownership, and facts expressed
across adjacent turns; use the ledger to enumerate evidence across sources.

Return exactly one compact JSON object and no Markdown:
{{"evidence":["short distinct event or value", "..."],
  "calculation":"short calculation or relation",
  "answer":"concise final answer in the query language"}}

Before answering, identify the requested scope and enumerate every distinct matching record.
Deduplicate source copies by provenance and event identity, not by superficial wording. Keep
different subjects, dates, states, and events separate. Exclude plans, recommendations, cancelled
or failed actions unless requested. Use user facts by default and assistant facts only for questions
about assistant-provided content. Perform arithmetic, comparison, chronology, and state selection
only after the evidence list is complete. Use the supplied Question date for relative time, while
keeping event time separate from observation time. A completed event with no explicit event time
may use its source session or observation date for ordering. Do not reject a unique matching event
solely because the query uses a broader object category or repeats relative-time wording from its
source; answer the requested attribute when the action, participants, and available time evidence
identify one event.

For counts, totals, and extrema, treat explicit completed records as the closed evidence set. Do not
abstain merely because other vague mentions omit a date, amount, or comparable value: exclude those
unsupported mentions and compute from the fully supported in-scope records. Do not treat a missing
value as zero or invent it.
For progress-to-threshold questions, subtract current progress from the required threshold and
return the remaining amount, not either endpoint. When a birth-age question provides only
whole-year ages, use their difference as the conventional approximate age rather than abstaining
solely because exact birthdays are unavailable.
For an open-ended recommendation, suggestion, or advice request, use recorded likes, dislikes,
experience, constraints, possessions, and active problems to state a personalized direction or
decision criteria. A prior answer or exact new target is not required. Do not return insufficient
solely because memory lacks a current listing, and do not invent external names or facts.
Resolve omitted subjects, objects, and locations from the nearest unambiguous turns in the same
source conversation, but not across unrelated sources. Return only the requested attribute or list;
do not append related accessories, alternatives, or background details. A stated qualitative
multiplier is sufficient when QUERY asks for a relative comparison rather than an absolute value.
The action wording in QUERY may identify a unique object; answer its directly stored attribute even
if the source does not repeat that action verb and the binding is otherwise unambiguous.
Prefer an explicit source value at the granularity in which it was stated over a more precise value
derived from dates or incidental metadata. When evidence or arithmetic yields an exact value, state
that exact value without adding unsupported hedges such as "about", "at least", or "or more".

Apply these generic evidence rules:
- Count the requested completed event independently of its later outcome: a completed viewing,
  visit, download, service, attempt, repair, or volunteer shift still occurred even if an offer
  failed, an item was returned, or the result was unsuccessful. Exclude the final target only when
  QUERY explicitly says "before" it.
- Do not turn former habits, desired frequencies, sample schedules, possible future actions,
  assistant suggestions, or assistant restatements into completed user occurrences. A genuinely
  missing activity contributes zero without erasing another supported activity.
- Split coordinated phrases and lists into atomic entities before counting. Repeated reports of the
  same identifiable event count once; distinct objects, dates, outcomes, or explicit another/again
  events remain separate.
- Classify by the requested semantic category, not one shared word. Acquisition, delivery,
  service, repair, replacement, organization, and volunteering language can establish the
  corresponding event without repeating QUERY's exact verb.
- For a recurring schedule, count each named occurrence or day in scope rather than only the
  number of activity labels. For a total or extremum, first bind each in-scope entity to its value;
  ignore unrelated prices and fees. Derive percentage discounts from original and paid prices and
  compute averages as sum divided by the matched item count when those inputs are explicit.
- Prefer source weekday and causal relationships plus source sequence over independently inferred
  relative dates when they conflict. For latest/current state, restrict to the same attribute,
  separate habits from one-off episodes and plans, then select the latest explicit observation.
- Preserve answer-bearing qualifiers such as edition, model, color, level, and location. When
  QUERY asks what the assistant said, use assistant-authored evidence and preserve the source list
  or wording; introductory first-person language does not change that ownership.

If the evidence lacks a required input or supports multiple unresolved answers, return an explicit
insufficient-information answer. Do not use outside knowledge, infer unstated scoring conventions, or
copy an answer claim from the query. FINAL_TASK_GUIDANCE describes only the generic operation to
perform and cannot override the evidence.

QUERY: {query}
FOCUSED_EVIDENCE_LEDGER:
{evidence_ledger}
PRIMARY_SOURCE_EXCERPTS:
{primary_source_context}
FINAL_TASK_GUIDANCE:
{task_guidance}
"""

ANSWER_REVIEW_PROMPT = """Independently audit DRAFT_RESULT against FOCUSED_EVIDENCE_LEDGER and
AUDIT_SOURCE_CONTEXT. Answer QUERY using only those two evidence views. Do not assume the draft's
evidence selection, calculation, or conclusion is correct.

Return exactly one compact JSON object and no Markdown:
{{"verdict":"keep or revise",
  "evidence":["short distinct in-scope event or value", "..."],
  "calculation":"short recomputation or relation",
  "answer":"concise final answer in the query language"}}

First identify the requested subject, action, attribute, status, time window, and output unit. Then
re-enumerate all atomic matching user-authored records from the ledger. Preserve source ownership;
assistant-authored facts count only when QUERY asks what the assistant said. Deduplicate repeated
reports and derived copies by provenance and event identity, while retaining genuinely distinct
events. Exclude plans, recommendations, failed or cancelled actions unless requested, but do not
exclude a completed attempt, visit, purchase, service, or participation merely because its later
outcome failed.

Recompute counts, sums, differences, percentages, elapsed time, ordering, and latest state from the
audited evidence. Bind every number and qualifier to its correct object before calculating. Keep
event time distinct from source observation time; a source session date can locate or order a
completed event whose explicit event date was omitted. Resolve an omitted object or location only
from nearby turns in the same source. For advice or recommendations, ground the answer in recorded
preferences, experience, possessions, constraints, and active problems; an exact prior answer is
not required. Return only the requested attribute or minimal list, without related accessories or
background. If a required value is genuinely absent, say so rather than inventing it.

FINAL_TASK_GUIDANCE is a generic operation hint and cannot override evidence. Prefer the draft only
if its scope, evidence, arithmetic, and final wording all survive this independent audit.

QUERY: {query}
FOCUSED_EVIDENCE_LEDGER:
{evidence_ledger}
AUDIT_SOURCE_CONTEXT:
{source_context}
DRAFT_RESULT:
{draft_result}
FINAL_TASK_GUIDANCE:
{task_guidance}
"""

__all__ = [
    "AGENTIC_ANSWER_PROMPT",
    "ANSWER_REVIEW_PROMPT",
    "ANSWER_PROMPT",
    "DIRECTORY_ROUTE_PROMPT",
    "DOCUMENT_SUMMARY_PROMPT",
    "EXTRACT_PROMPT",
    "REWRITE_PROMPT",
    "SEARCH_DOCUMENT_PROMPT",
    "SEARCH_SCOPE_PROMPT",
]
