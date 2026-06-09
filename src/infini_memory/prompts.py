SYSTEM_DEFAULT = "You are InfiniMemory, a helpful assistant."

# Memory Extraction Prompt: extract user-related facts/preferences from conversations, output Markdown
EXTRACT_MEMORY_PROMPT = """You are a personal information organizer dedicated to accurately storing facts, user memories, and preferences.
Extract relevant information fragments from the conversation and organize them as clear, manageable independent facts
for future retrieval and personalized use in interactions.

**Input may contain**:
- User input (User): user statements, questions, preferences, etc.
- AI replies (Assistant): AI replies may contain confirmed user information, suggestions, summaries, etc.

Types of information to focus on (examples, including but not limited to):
1) Personal preferences (likes/dislikes): food, products, activities, entertainment, etc.;
2) Important personal information: names, relationships, important dates;
3) Plans and intentions: events, travel, goals, and any plans;
4) Activity and service preferences: dining, travel, hobbies, and other service preferences;
5) Health and lifestyle: dietary restrictions, fitness habits, and other health-related information;
6) Career-related: job title, work habits, career goals;
7) Miscellaneous information: favorite books, movies, brands, and other details.

**Important: Processing information from AI replies**
- If an AI reply confirms, summarizes, or mentions user information (e.g., "OK, I'll remember you like coffee", "You mentioned earlier that you live in New York"), this information should also be extracted
- When extracting information from AI replies, you must mark the source in the metadata: `source=AI`
- Example:
  - User says: "I like coffee"
  - AI replies: "OK, I'll remember you like coffee, next time I'll recommend relevant coffee shops"
  - Extract as: `- <seq=@@SEQ@@,time=empty> Likes coffee` (from user, no source annotation needed)
  - Extract as: `- <seq=@@SEQ@@,time=empty,source=AI> Likes coffee` (from AI confirmation, must annotate source=AI)

**Important: Preserve metadata from the original text (e.g., label)**
- If the original text contains metadata tags like `label: xxx`, they must be preserved inside `< >`
- Format: `<seq=@@SEQ@@,time=timestamp,label=xxx,...>` (add label and other metadata inside angle brackets)
- Example:
  - Original: "What is the procedure to transfer money to my account? label: 52"
  - Extract as: `- <seq=@@SEQ@@,time=empty,label=52> What is the procedure to transfer money to my account?`
  - Original: "I can't input my pin again. label: 0"
  - Extract as: `- <seq=@@SEQ@@,time=empty,label=0> I can't input my pin again.`
- If the original text has no label or other metadata, do not add label fields

**Core principles**:
1. **Absolute faithful recording**: Extracted facts must be completely faithful to the source text, recording verbatim what the user said
2. **No annotations allowed**: Strictly prohibited from adding any form of corrections, annotations, parenthetical notes, or error-pointing text
3. **No questioning content**: Regardless of whether the original content seems reasonable, accurate, or truthful, it must be recorded as-is without questioning or modification
4. **No illustrative explanations**: Do not add "(Note: this is...)", "(e.g., ...)" or similar explanatory text
5. **Incorrect examples** (strictly avoid):
   - BAD: Hugh Lofting's music genre is children's literature (Note: this is a misclassification, Hugh Lofting is a writer, not a musician)
   - GOOD: Hugh Lofting's music genre is children's literature
   - BAD: User says they live in New York (but it might actually be New Jersey)
   - GOOD: User says they live in New York

**Sequence number placeholder mandatory rule (extremely important)**:
- All sequence numbers must be output **verbatim** as: `<seq=@@SEQ@@,...>`
- `@@SEQ@@` is a **plain text placeholder**, not a variable, not an example, not content to be filled in
- **Strictly prohibited** from inferring, generating, replacing, incrementing, simulating, or guessing any sequence numbers
- Even if you "think" a number should be filled in, you must keep `@@SEQ@@` unchanged

Output format requirements (Markdown):
- You should automatically detect the language of the user's input and use the same language for the output document;
- Add YAML Frontmatter at the beginning of the document, in the following format:
  ---
  summary: <Format: "keyword1, keyword2, ...; concise factual summary of the content". Keywords should capture the core categories of the document (e.g., career, diet, travel, health, etc.), separated by commas. Keywords and summary are separated by semicolon. The summary should include entities, facts, numbers, timestamps, and other key information. Max length: {summary_length} tokens.>
  ---
- Output Markdown content only (including YAML Frontmatter), no explanations or system instructions;
- After the YAML Frontmatter, organize the body with first-level headings (e.g., "# Personal Preferences", "# Plans and Intentions", etc.), headings are examples only and can be customized based on content;
- **Each fact must begin with a sequence number and timestamp**, in the format:
  - User information: `<seq=@@SEQ@@,time=timestamp>` (no source annotation needed)
  - User information (with label): `<seq=@@SEQ@@,time=timestamp,label=xxx>` (preserve original label)
  - AI information: `<seq=@@SEQ@@,time=timestamp,source=AI>` (must annotate source=AI)
  - AI information (with label): `<seq=@@SEQ@@,time=timestamp,label=xxx,source=AI>` (preserve label and annotate source)
- **Source annotation rules**:
  - If information comes from user input, no source annotation needed, format: `<seq=@@SEQ@@,time=timestamp>` or `<seq=@@SEQ@@,time=timestamp,label=xxx>`
  - If information comes from AI reply, must annotate source=AI, format: `<seq=@@SEQ@@,time=timestamp,source=AI>` or `<seq=@@SEQ@@,time=timestamp,label=xxx,source=AI>`
- **Metadata preservation rules (extremely important)**:
  - If the original text contains `label: xxx` (or other `key: value` format metadata), it must be preserved as-is inside `< >`
  - Format: add metadata fields after seq, time, such as `label=52`, `category=xxx`, etc.
  - Strictly prohibited from omitting or modifying label or other metadata from the original text
  - Example:
    - Original: "The ATM did not give me my card back? label: 51"
    - Extract as: `- <seq=@@SEQ@@,time=empty,label=51> The ATM did not give me my card back?`
- **Timestamp precision rules**:
  1. If there is time information from conversation records or the user explicitly mentions a time, use that time, precise to seconds (format: YYYY-MM-DD HH:MM:SS);
  2. If only a date without time-of-day, use the date (format: YYYY-MM-DD);
  3. If no time information is available, use time=empty; do not use the current system time;
- Each fact uses an unordered list (`- <seq=@@SEQ@@,time=2022-06-15> fact content`), kept brief and independent;
- If a fact contains a label, format: `- <seq=@@SEQ@@,time=2022-06-15,label=52> fact content`;
- If no extractable facts exist, output:
  ---
  summary: No memorable facts
  ---
  # No memorable facts

  - No memorable facts

**Pre-output self-check (must be performed)**:
- Check whether any non-`@@SEQ@@` sequence number content exists (such as numbers or other text)
- If found, treat as a critical error and correct to `@@SEQ@@`
- If the original text contains `label: xxx`, verify it has been correctly preserved as `label=xxx` format
- Ensure all metadata is kept inside `< >` angle brackets, not left in the body text
"""


# CURRENT Document Rewrite Prompt: aggregate rewrite of append-only CURRENT document by topic
REWRITE_CURRENT_PROMPT = """You will receive the content of a CURRENT document (this document is append-only and may contain facts from multiple mixed topics).

Your task is:
**Without introducing any new information, explanations, or opinions,
aggregate and reorganize the facts in the CURRENT document by topic, producing a clearly structured Markdown document.**

━━━━━━━━━━━━━━━━━━━━
[HIGHEST PRIORITY CONSTRAINTS (must be strictly followed)]
━━━━━━━━━━━━━━━━━━━━

1. **Absolute factual faithfulness**
   - You may only reorganize, categorize, and deduplicate facts that explicitly appear in the original text
   - Strictly prohibited from supplementing, inferring, generalizing, weakening, or strengthening any facts
   - Even if a piece of content is obviously wrong, unreasonable, or contradictory, it must be preserved as-is (only choose which to keep under the conflict rules)

2. **No explanatory output allowed**
   - No annotations, notes, remarks, corrections, or background introductions of any kind
   - No parenthetical supplementary information
   - Do not use expressions like "Note:", "e.g.:", "possibly", "it seems", "actually"

3. **No questioning or judgment**
   - Do not make any judgment about the truthfulness, reasonableness, or accuracy of facts
   - Do not imply that facts may be incorrect

4. **No illustrative or pedagogical language**
   - Output must be "recorded results", not "process explanation"

> **Any action not explicitly permitted by the above rules is prohibited.**

━━━━━━━━━━━━━━━━━━━━
[AGGREGATION AND REWRITE RULES (only permitted operations)]
━━━━━━━━━━━━━━━━━━━━

1. **Aggregate by topic**
   - Group facts that semantically belong to the same topic together
   - Do not change the meaning of any fact

2. **Sequence numbers, timestamps, and metadata must be fully preserved**
   - User fact format:
     `<seq=sequence_number,time=timestamp>` or `<seq=sequence_number,time=timestamp,label=xxx,...>`
   - AI fact format:
     `<seq=sequence_number,time=timestamp,source=AI>` or `<seq=sequence_number,time=timestamp,label=xxx,source=AI,...>`
   - **All metadata fields must be preserved as-is** (e.g., label, category, etc.), do not modify, add, or omit

3. **Deduplication rules**
   - For semantically identical duplicate facts, keep only one
   - If multiple versions exist, keep the one with the larger `seq` number

4. **Contradictory fact handling**
   - If facts contradict each other but have different `seq` values:
     - Keep only the fact with the larger `seq`
     - Do not note or annotate that "a replacement occurred"

5. **Structural organization**
   - Use Markdown first-level headings (`#`) for topics
   - Headings are for topic grouping only, must not contain summarizing or evaluative language

6. **Language consistency**
   - Automatically detect the original language
   - Output must use the same language, do not mix or translate

━━━━━━━━━━━━━━━━━━━━
[OUTPUT REQUIREMENTS]
━━━━━━━━━━━━━━━━━━━━

- Output only the final Markdown document
- Do not output any explanations, hints, analysis, or system text

━━━━━━━━━━━━━━━━━━━━
[CURRENT DOCUMENT CONTENT]
━━━━━━━━━━━━━━━━━━━━

{current_content}
"""



# Update Plan Prompt: given candidate document summaries and new Markdown content, output documents to update and new documents to create
PLAN_UPDATE_PROMPT = """You will receive: 1) new Markdown content; 2) existing document list (id + summary).

━━━━━━━━━━━━━━━━━━━━
[TASK]
━━━━━━━━━━━━━━━━━━━━
- Determine which existing documents need to be updated, and provide an update fragment list aggregated by document id;
- If there is information that cannot reasonably be assigned to any existing document, organize it into one or more new documents (split by topic).

━━━━━━━━━━━━━━━━━━━━
[HIGHEST PRIORITY CONSTRAINTS (must be strictly followed)]
━━━━━━━━━━━━━━━━━━━━

1. **Sequence numbers, timestamps, and metadata must be fully preserved**
   - User fact format must be maintained: `<seq=sequence_number,time=timestamp>` or `<seq=sequence_number,time=timestamp,label=xxx,...>`
   - AI fact format must be maintained: `<seq=sequence_number,time=timestamp,source=AI>` or `<seq=sequence_number,time=timestamp,label=xxx,source=AI,...>`
   - **Do not modify, supplement, omit, or regenerate any seq, time, source, label, or other fields**
   - Renumbering or rewriting timestamps or metadata is prohibited

> **Any action not explicitly permitted by the above rules is prohibited.**

━━━━━━━━━━━━━━━━━━━━
[SPLITTING REQUIREMENTS (must be followed)]
━━━━━━━━━━━━━━━━━━━━

1. **Cluster then split**: Before splitting, first cluster the new content by topic, grouping similar topics together, then create one new document for each topic group;
2. **Mandatory splitting**: Each new document's token count **must be strictly kept within {markdown_length}** (hard limit, cannot be exceeded);
3. **Multi-document splitting**: If the new content exceeds {markdown_length} tokens, you **must** split it into multiple new documents, not generate a single long document;
4. **Topic granularity**: When splitting, ensure each document is topically focused while keeping the total number of documents reasonable (avoid excessive fragmentation, also avoid single overly long documents);

━━━━━━━━━━━━━━━━━━━━
[OUTPUT FORMAT]
━━━━━━━━━━━━━━━━━━━━

Output JSON only, do not include any other content.

Output JSON schema: {{
  "updates": [{{"id": string, "new_content": string}}],
  "new_docs": [{{"title": string, "content": string}}, ...]
}}

Important notes:
1. If the existing document list is empty (docs is an empty array), you must split all new content by topic into multiple new documents in the new_docs list; do not return an empty result.
2. new_docs is a list that can contain multiple new documents (split by topic).
3. Each new document should focus on a single clear topic; different topics should be split into separate documents.
4. **No content overlap between generated new documents**: Ensure each fact appears in only one new_doc or update, avoiding the same content being assigned to multiple documents.
5. **updates IDs must come from the input docs list**: Strictly prohibited from fabricating, truncating, or modifying IDs; if no existing ID can be matched, use new_docs instead.
"""


# Document Rewrite Prompt: rewrite a document based on old document and new content to append (without changing the id)
REWRITE_DOC_PROMPT = """You will receive: 1) an old Markdown document; 2) new content to be merged.

Your task is:
**Without introducing any new information, explanations, or opinions,
produce the updated complete Markdown document.**

━━━━━━━━━━━━━━━━━━━━
[HIGHEST PRIORITY CONSTRAINTS (must be strictly followed)]
━━━━━━━━━━━━━━━━━━━━

1. **Sequence numbers, timestamps, and metadata must be fully preserved**
   - User fact format must be maintained: `<seq=sequence_number,time=timestamp>` or `<seq=sequence_number,time=timestamp,label=xxx,...>`
   - AI fact format must be maintained: `<seq=sequence_number,time=timestamp,source=AI>` or `<seq=sequence_number,time=timestamp,label=xxx,source=AI,...>`
   - **Do not modify, supplement, omit, or regenerate any seq, time, source, label, or other fields**
   - **Renumbering, rewriting timestamps, deleting original annotations or metadata is prohibited**
   - Do not replace original seq values with new ones

> **Any action not explicitly permitted by the above rules is prohibited.**

━━━━━━━━━━━━━━━━━━━━
[MERGE AND REWRITE RULES (only permitted operations)]
━━━━━━━━━━━━━━━━━━━━

1. **Content source**
   - Facts can only be drawn from the old document and the new content
   - Do not add information that does not exist in the original texts

2. **Structural organization**
   - Use Markdown first-level headings (`#`) for topics
   - Headings are for topic grouping only, must not contain summarizing or evaluative language
   - Keep the structure clear

3. **Language consistency**
   - Automatically detect the language of the old document
   - Output must use the same language, do not mix or translate

4. **Contradictory fact handling rules (by priority, highest first)**
   - **Contradictions within the same add() call**: If there are contradictory facts within the new content, the later content in the document takes precedence; remove the earlier contradicting content;
   - **Contradictions across add() calls**: If contradictory facts exist between old and new content (different sequence numbers), the one with the larger sequence number takes precedence (keep the larger seq fact, delete the smaller seq contradicting fact);
   - Criteria for determining contradictions: the same item has opposite, mutually exclusive, or incompatible descriptions (e.g., "likes coffee" vs "doesn't like coffee", "lives in New York" vs "lives in Los Angeles", "married" vs "single");
   - **Source priority**: If two facts share the same sequence number, one from the user (no source) and one from AI (source=AI), the user input takes precedence (keep the version without source);

━━━━━━━━━━━━━━━━━━━━
[OUTPUT FORMAT REQUIREMENTS]
━━━━━━━━━━━━━━━━━━━━

Add YAML Frontmatter at the beginning of the document, in the following format:
---
summary: <Format: "keyword1, keyword2, ...; concise factual summary of the content". Keywords should capture the core categories of the document (e.g., career, diet, travel, health, etc.), separated by commas. Keywords and summary are separated by semicolon. The summary should include entities, facts, numbers, timestamps, and other key information. Max length: {summary_length} tokens.>
---

Each fact uses an unordered list, preserving the original format:
- User fact: `<seq=sequence_number,time=timestamp> fact content` or `<seq=sequence_number,time=timestamp,label=xxx> fact content`
- AI fact: `<seq=sequence_number,time=timestamp,source=AI> fact content` or `<seq=sequence_number,time=timestamp,label=xxx,source=AI> fact content`

Output Markdown content only (including YAML Frontmatter), no explanations or system instructions.

━━━━━━━━━━━━━━━━━━━━
[INPUT CONTENT]
━━━━━━━━━━━━━━━━━━━━

[OLD DOCUMENT]
{old_content}

[NEW CONTENT]
{new_content}

Output: Only output the updated complete Markdown document content (including YAML Frontmatter), do not include any other content, explanations, or JSON format."""


# Select Merge Groups Prompt: select groups of topically similar documents from the existing library for merging
SELECT_MERGE_GROUPS_PROMPT = """You will receive an existing document list (id + summary + update time).
Task: Analyze topic similarity between documents, select groups of topically similar documents, 2-4 documents per group, for subsequent merging.

Selection criteria:
1. Highly related topics: Documents discuss the same or similar topics (e.g., all about travel, all about reading, etc.)
2. Content overlap: Documents have information that can be consolidated or deduplicated
3. Moderate group size: 2-4 documents per group, avoid too many or too few
4. Priority merging: Prioritize grouping documents with high summary similarity

**Most important constraint (must be strictly followed)**:
- Each document can only appear in one group; the same document appearing in multiple groups is strictly prohibited
- Groups must not overlap; once a document is selected, it cannot be used in another group
- Groups should be independent, mutually exclusive sets of documents

Other constraints:
- If fewer than 2 documents exist, return an empty group list
- If no topically similar documents exist, return an empty group list
- Automatically detect document language and use the same language for output

Output JSON schema: {{
  "groups": [
    {{"doc_ids": [string, ...], "reason": "explanation for grouping"}},
    ...
  ]
}}

Output example:
{{
  "groups": [
    {{"doc_ids": ["1_2025-01-01_abc", "1_2025-01-02_def"], "reason": "Both documents are about travel plans and airline mileage management"}},
    {{"doc_ids": ["1_2025-01-03_ghi", "1_2025-01-04_jkl", "1_2025-01-05_mno"], "reason": "All three documents involve reading preferences and book reviews"}}
  ]
}}

**Self-check steps (must be performed before output)**:
1. Check all doc_ids lists to ensure no document ID appears in multiple groups
2. If duplicates are found, readjust groups to ensure mutual exclusivity
3. Ensure groups are completely independent with no shared documents

Output JSON only, do not include any other content."""


# Merge Multiple Documents Prompt: merge the content of multiple documents into a single new document
MERGE_DOCS_PROMPT = """You will receive the content of multiple Markdown documents.
Task: Merge all documents into one complete, coherent Markdown document.

Merge rules:
1. **Sort by update time**: Sort documents by updated_at time from oldest to newest, with newer document content placed later
2. **Preserve all sequence numbers and metadata**: You must preserve each fact's original sequence number, timestamp, and metadata format without modification or deletion:
   - User information: `<seq=sequence_number,time=timestamp>` or `<seq=sequence_number,time=timestamp,label=xxx,...>` (no source)
   - AI information: `<seq=sequence_number,time=timestamp,source=AI>` or `<seq=sequence_number,time=timestamp,label=xxx,source=AI,...>` (with source=AI)
   - **All metadata fields must be preserved as-is** (e.g., label, category, etc.)
3. **Deduplication**: Remove completely duplicate facts, keeping the version with the larger sequence number
4. **Contradictory fact handling**: If contradictory facts exist (different sequence numbers), the one with the larger sequence number takes precedence
5. **Structural optimization**: Use first-level headings to organize content for a clear document structure
6. **Language preservation**: Automatically detect document language and use the same language for output
7. **Source priority**: If completely identical facts exist (same content, sequence number, and timestamp) but with different sources (one without source, one with source=AI), keep only the version without source (user information takes priority)

**Core principles**:
1. **Absolute faithful recording**: Extracted facts must be completely faithful to the original text, recording verbatim what the user said
2. **No annotations allowed**: Strictly prohibited from adding any form of corrections, annotations, parenthetical notes, or error-pointing text
3. **No questioning content**: Regardless of whether the original content seems reasonable, accurate, or truthful, it must be recorded as-is without questioning or modification
4. **No illustrative explanations**: Do not add "(Note: this is...)", "(e.g., ...)" or similar explanatory text
5. **Incorrect examples** (strictly avoid):
   - BAD: Hugh Lofting's music genre is children's literature (Note: this is a misclassification, Hugh Lofting is a writer, not a musician)
   - GOOD: Hugh Lofting's music genre is children's literature
   - BAD: User says they live in New York (but it might actually be New Jersey)
   - GOOD: User says they live in New York

Output format requirements:
- Add YAML Frontmatter at the beginning of the document, in the following format:
  ---
  summary: <Format: "keyword1, keyword2, ...; concise factual summary of the content". Keywords should capture the core categories of the document (e.g., career, diet, travel, health, etc.), separated by commas. Keywords and summary are separated by semicolon. The summary should include entities, facts, numbers, timestamps, and other key information. Max length: {summary_length} tokens.>
  ---
- Output Markdown content only (including YAML Frontmatter), no explanations or system instructions;
- Organize content using first-level headings (e.g., "# Personal Preferences", "# Plans and Intentions", etc.), headings are examples only and can be customized based on content;
- **Each fact must preserve its original sequence number, timestamp, and metadata format**:
  - User information: `<seq=sequence_number,time=conversation_time>` or `<seq=sequence_number,time=conversation_time,label=xxx,...>` (no source)
  - AI information: `<seq=sequence_number,time=conversation_time,source=AI>` or `<seq=sequence_number,time=conversation_time,label=xxx,source=AI,...>` (with source=AI)
- Each fact uses an unordered list (`- <seq=1,time=2025-01-15> fact content`) or (`- <seq=2,time=2025-01-15,source=AI> fact content`), kept brief and independent;
- If time information exists, record the full date (e.g., January 15, 2025).

[MULTIPLE DOCUMENT CONTENT]
{docs_content}

Output: Only output the merged complete Markdown document content (including YAML Frontmatter), do not include any other content, explanations, or JSON format."""


# Memory Search Prompt: select relevant documents based on query and document list (id + summary), output JSON id list
SEARCH_MEMORY_PROMPT = """You will receive: 1) a user query; 2) an existing document list (id + summary).
Task: Select documents relevant to the query.
Return no more than {search_limit} results.
Output JSON containing only an id list: {{"ids":[string,...]}}. Do not output anything else."""


# Answer with Context Prompt: answer questions based on retrieved results
ANSWER_WITH_CONTEXT_PROMPT = """You are an expert at answering questions based on provided content. Your task is to provide accurate, concise answers to questions by leveraging the provided information.

Guidelines:
- Extract relevant information from the provided content based on the question;
- If no relevant information is found, do not say "no information found". Instead, accept the question and provide a general response;
- Ensure the answer is clear, concise, and directly addresses the question.

Task details will be provided via user message in JSON format: {{"query": "<user query>", "context": "<relevant document content (possibly multiple, full Markdown format, including YAML Frontmatter metadata)>"}}.

Additional requirements:
- Automatically detect the language of the user query and use the same language for the answer;
- Be objective and concise; do not fabricate details;
- Document content includes metadata in YAML Frontmatter (such as summary); you can reference this information to understand the document context;
- If there are multiple documents, synthesize them before answering.

**About sequence numbers and timestamp format in documents**:
- Each fact begins with a sequence number and timestamp, in the format:
  - User information: `<seq=sequence_number,time=timestamp>` (no source annotation)
  - AI information: `<seq=sequence_number,time=timestamp,source=AI>` (with source=AI annotation)
- `seq` is the sequence number, **used only to indicate the chronological order of information recording**; larger numbers indicate newer information;
  - **The sequence number is NOT a category label, NOT a category ID, NOT an answer number**
  - **Strictly prohibited from outputting the sequence number as a question answer, category, or label**
- `time` is the timestamp, which may be in the format:
  - Full time: `2025-01-15 14:30:00` or `2025-01-15`
  - `time=empty` means the original text had no explicit time information;
- `source=AI` indicates the information was extracted from an AI reply (such as AI confirmation, summary, mention of user information); absence of this annotation means the information came directly from user input;
- When answering questions, you may reference sequence numbers and timestamps to determine information recency and source, prioritizing newer information.

Output: Return only the final answer text."""


# Evaluation Judge Prompt: determine whether the model answer and expected answer are semantically consistent
# Requirement: output only Yes or No (case-insensitive, recommend Yes/No)
EVAL_JUDGE_PROMPT = """You will be given an original question (Question), a model answer (Model Answer), and an expected answer (Expected Answer).
Based on the original question, determine whether the model answer correctly or reasonably answers the question.

Judgment principle: As long as the model answer is semantically correct or partially correct, it should be judged as Yes. Only when the model answer is clearly wrong, completely irrelevant, or severely missing key information should it be judged as No. When in doubt, judge Yes.

Judgment rules (meeting any one condition results in Yes):
1. Semantic equivalence or similarity: The model answer is semantically equivalent or similar to the expected answer, even if wording or level of detail differs. For example: "direct flight" vs "Non-stop round-trip economy", "accumulating airline miles for travel rewards" vs "The user is motivated by the large sign-up bonus and travel-related benefits";
2. Contains key information: The model answer contains the core or key information from the expected answer (such as numbers, entities, facts, motivations, etc.), even if not detailed enough or missing secondary information;
3. Extension of correct answer: The model answer adds additional reasonable information on top of the correct answer (e.g., "Two" vs "Two free nights", "3" vs "3 months ago");
4. Concrete answer: When the expected answer is descriptive text (e.g., "the user would prefer..."), if the model answer gives a specific suggestion or answer that matches the description, it should be judged as Yes. For example: expected answer describes "the user would prefer baking suggestions considering the previous successful lemon poppy seed cake", model answer "lemon lavender cake" should be judged as Yes;
5. Equivalent expression: The model answer uses different language (Chinese-English interchange), units, expressions, or formats, but the core information conveyed is the same. For example:
   - Pronoun interchange: "Your sister" vs "my sister", "your mom" vs "my mom" (you/I, your/my should be considered equivalent)
   - Unit differences: "20 dozen" vs "20", "3 kg" vs "3", "5 miles" vs "5" (units can be omitted or added)
   - Format differences: case, punctuation, and other format differences do not affect semantics
   - Different expressions for "insufficient information": "Zero - none mentioned in the chat history" vs "The information provided is not enough" vs "Cannot obtain this information from the chat history" (different ways of expressing insufficient information should be considered equivalent)
6. Reasonable simplification: The model answer reasonably simplifies or summarizes the expected answer while preserving the core meaning;
7. Direct answer to question: When the expected answer is descriptive text (describing user preferences or expected answer style), if the model answer directly provides a specific answer or suggestion that fits the description, it should be judged as Yes. In this case, the model answer is reasonable and useful, even if the format differs from the expected answer;
8. Partial list match: When the expected answer is a list or enumeration (multiple items, reasons, or examples), the model answer is correct if it includes at least the majority of the key items. Missing one or two less important items does not warrant No. For example: expected "apples, bananas, oranges, grapes" and model "apples, bananas, and oranges" should be Yes;
9. Correct conclusion with different reasoning: For fact-checking or verification questions, if the model answer reaches the same factual conclusion as the expected answer (e.g., both confirm or deny a fact), it should be judged as Yes regardless of whether the reasoning path, evidence cited, or supporting details differ;
10. Correctly inferred information: If the model answer provides information that is logically implied by or can be reasonably inferred from the expected answer, even if not stated word-for-word, it should be judged as Yes. For example: expected "the user moved to Tokyo in March", model "the user lives in Tokyo" is Yes.

Important notes:
- Expected answers are sometimes specific answers, sometimes descriptive text (describing "what type of answer the user would prefer")
- When the expected answer is descriptive text, if the model answer provides a specific suggestion that fits the description, it should be judged as Yes
- Do not judge as No simply because the model answer is more concise or has a different format
- Differences in pronouns (you/I, your/my), units (dozen/kg/miles, etc.), case, punctuation, etc. should not affect the judgment

Conditions for judging No (ALL conditions must be met):
- The model answer's core factual claim directly contradicts the expected answer, OR the model answer is completely irrelevant to the question;
- AND the model answer does not contain any of the key facts, entities, or conclusions present in the expected answer.

When in doubt, judge Yes. A partial answer or an answer with minor inaccuracies is still Yes.

Input will be provided in JSON: {{"question": string, "model_answer": string, "expected_answer": string}}.
Output: Return only Yes or No, do not output anything else."""


# InfBench_sum Summary Evaluation Prompts

# Fluency evaluation prompt
SUMM_FLUENCY_PROMPT = """Please act as an impartial judge and evaluate the fluency of the provided text. The text should be coherent, non-repetitive, fluent, and grammatically correct.

Below is your grading rubric:
- Score 0 (incoherent, repetitive, or incomplete): Incoherent sentences, repetitive sentences (even if not by exact words), incomplete answers, or gibberish. Note that even if the answer is coherent, if it is repetitive or incomplete, it should be given a score of 0.

- Score 1 (coherent, non-repetitive answer): Coherent, non-repetitive, fluent, grammatically correct answers. If the text is coherent, non-repetitive, and fluent, but the last sentence is truncated, it should still be given a score of 1.

Now, read the provided text, and evaluate the fluency using the rubric. Then output your score in the following json format: {{"fluency": 1}}.

Text: "{text}"
"""


# Recall evaluation prompt (for novels/books)
SUMM_RECALL_PROMPT = """Please act as an impartial judge and evaluate the quality of the provided summary of a novel. It should discuss the plots and characters of the story. The text should contain all the given key points.

Below is your grading rubric:
Recall:
- Evaluate the provided summary by deciding if each of the key points is present in the provided summary. A key point is considered present (supported) if any of the following apply:
  - The summary explicitly states the key point's main fact.
  - The summary conveys the same information using different words, names, or phrasing.
  - The summary implies or alludes to the key point, even if not stated explicitly.
  - If a key point contains multiple facts, it is supported if at least half of the facts are present or implied in the summary.
- When in doubt about whether a key point is supported, lean toward marking it as supported. Minor differences in wording, level of detail, or emphasis do not make a key point unsupported.
- Score: the number of key points supported by the provided summary.

Now, read the provided summary and key points, and evaluate the summary using the rubric. First, think step-by-step and provide your reasoning and assessment on the answer. Then output your score in the following json format: {{"supported_key_points": [2, 4], "recall": 2}}, where "supported_key_points" contains the indices of key points that are present in the summary and "recall" is the total number of key points present in the summary.

Key points:
{keypoints}

Summary: <start of summary>{summary}<end of summary>
"""


# Precision evaluation prompt (for novels/books)
SUMM_PRECISION_PROMPT = """Please act as an impartial judge and evaluate the quality of the provided summary of a novel.

Below is your grading rubric:
Precision:
- Evaluate the provided summary by deciding if each sentence in the provided summary is supported by the information provided in the expert summary.
- A sentence is considered SUPPORTED if any of the following apply:
  - Its major facts align with the information in the expert summary, even if minor details differ.
  - It makes reasonable inferences or connections between events described in the expert summary.
  - It provides thematic observations or high-level analysis that is consistent with the expert summary's content.
  - It is a transitional sentence, introduction, or conclusion that frames the story content covered by the expert summary.
  - Its content is implied by or logically follows from the expert summary, even if not stated explicitly.
- A sentence is NOT supported only if:
  - Its major facts are directly contradicted by the expert summary, OR
  - It describes events, characters, or plotlines about a completely different topic that the expert summary does not cover at all.
- Score: the number of sentences in the provided summary that are supported by the expert summary.

Now, read the provided summary and expert summary, and evaluate the summary using the rubric. First, think step-by-step and provide your reasoning and assessment on the answer. Then output your score in the following json format: {{"precision": 7, "sentence_count": 20}}.

Expert summary: <start of summary>{expert_summary}<end of summary>

Provided summary: <start of summary>{summary}<end of summary>
"""


# Extract keypoints from expert summary prompt (for summ_qa evaluation)
SUMM_EXTRACT_KEYPOINTS_PROMPT = """Please act as an impartial judge and extract key points from the provided expert summary of a novel. The key points should be factual information about plots and characters of the story.

Below is your extraction rubric:
- Extract 5 to 7 key points from the expert summary. Aim for exactly 5 to 6 key points when possible.
- Each key point should capture a major plot development, character arc, or thematic element -- not a minor detail.
- Prefer broader key points that encompass multiple related facts over narrow ones about single details.
- Key points should capture the major events, character actions, and relationships in the story.
- Avoid overly specific facts (exact dates, minor character names, trivial details) as standalone key points.
- Focus on: major plot turns, key character decisions, important relationships, and story outcomes.

Now, read the expert summary and extract the key points. Output your result in the following json format: {{"keypoints": ["key point 1", "key point 2", ...]}}.

Expert summary: <start of summary>{expert_summary}<end of summary>
"""


# Agentic Retrieval: Query Analysis Prompt
AGENTIC_QUERY_ANALYSIS_PROMPT = """You are an intelligent retrieval planner. Your task is to analyze the user query and formulate a retrieval plan.

You will receive:
1) User query
2) Existing document list (id + summary)
3) Current time

Analyze the query and output a retrieval plan in JSON format.

Analysis dimensions:
1. **Query type assessment**:
   - Is this a simple factual query (direct retrieval is sufficient)
   - Does the query need to be decomposed into multiple sub-queries (e.g., compound questions, comparison questions)
   - Does it involve time-related information (e.g., "recent", "last week", "yesterday")

2. **Retrieval method selection**:
   - `llm_search`: Semantic retrieval based on document summaries, suitable for conceptual and topical queries
   - `bm25_search`: Full-text keyword retrieval, suitable for queries containing specific entities, names, numbers
   - `bm25_partition_search`: Keyword retrieval based on document sections, suitable for queries requiring precise location of document fragments

3. **Retrieval limit**: Return at most {search_limit} documents

Output JSON schema:
{{
  "sub_queries": [string, ...],
  "retrieval_methods": ["llm_search" | "bm25_search" | "bm25_partition_search", ...],
  "reasoning": string
}}

Rules:
- `sub_queries`: If the query does not need decomposition, return a single-element list containing the original query; if decomposition is needed, return 2-3 sub-queries
- `retrieval_methods`: Select at least one retrieval method; multiple methods can be selected, results will be merged and deduplicated
- `reasoning`: One sentence explaining why this plan was chosen

Output JSON only, do not output anything else."""


# Agentic Retrieval: Relevance Evaluation Prompt (Corrective RAG mode)
AGENTIC_RELEVANCE_EVAL_PROMPT = """You are a retrieval result quality evaluator. Your task is to assess whether retrieved documents are relevant to the query and decide whether further retrieval is needed.

You will receive:
1) User query
2) Retrieved document list (id + summary + content snippet)
3) Current iteration count and maximum iterations

Evaluate each document's relevance and decide on the next action.

Evaluation criteria:
- **Relevant**: Document content directly answers the query, or contains key information needed to answer the query
- **Partially relevant**: Document content is related to the query topic but may not be sufficient to fully answer it
- **Not relevant**: Document content is unrelated to the query

Output JSON schema:
{{
  "relevant_doc_ids": [string, ...],
  "needs_more_retrieval": boolean,
  "rewritten_query": string | null,
  "next_methods": ["llm_search" | "bm25_search" | "bm25_partition_search", ...] | null,
  "reasoning": string
}}

Rules:
- `relevant_doc_ids`: List of document IDs judged as "relevant" or "partially relevant"
- `needs_more_retrieval`: If current results are insufficient to answer the query and iterations remain, set to true
- `rewritten_query`: If `needs_more_retrieval` is true, provide a rewritten query (different angle/keywords for retrieval); otherwise null
- `next_methods`: If `needs_more_retrieval` is true, specify retrieval methods for the next round; otherwise null
- `reasoning`: One sentence explaining the evaluation conclusion

Output JSON only, do not output anything else."""


# Agentic Retrieval: Tool-Calling Agent Prompt (grep-style precise tools)
AGENTIC_TOOL_AGENT_PROMPT = """You are a search agent. Your task is to find ALL information relevant to the user's query from a personal memory store.

You will receive a document catalog (doc_id + summary) in the first message. Use it to identify promising documents before searching.

Available tools:

1. grep(pattern, limit, context_lines)
   - Regex search across ALL document content. Returns matching lines with line numbers and context.
   - pattern: regex pattern (Python re syntax, case-insensitive)
   - limit: max matches to return (default {grep_limit}, max {grep_limit})
   - context_lines: lines of context before/after each match (default {grep_context_lines}, max 5)
   - Returns: {{"total_matches": int, "matches": [{{"doc_id", "line_number", "matched_line", "context_before": [...], "context_after": [...]}}], "truncated": bool}}
   - Best for: finding specific text, names, numbers, dates across all documents
   - Tip: use simple patterns. When total_matches > limit (truncated=true), refine your pattern to be more specific.

2. grep_doc(doc_id, pattern, context_lines)
   - Regex search within a SINGLE document. Returns ALL matches (no limit).
   - doc_id: the document to search in
   - pattern: regex pattern (Python re syntax, case-insensitive)
   - context_lines: lines of context before/after each match (default {grep_context_lines}, max 5)
   - Returns: {{"total_matches": int, "matches": [{{"line_number", "matched_line", "context_before": [...], "context_after": [...]}}]}}
   - Best for: thorough search within a specific document after identifying it via grep or search.

3. search(query, limit)
   - BM25 keyword search over document sections (split by H1 headings).
   - Returns: [{{"doc_id", "partition_index", "partition_title", "score", "snippet", "total_lines"}}]
   - snippet: an excerpt (~{search_snippet_tokens} tokens) around the best-matching terms
   - Best for: topical keyword search when you want to find relevant sections

4. list_docs(offset, limit)
   - Browse the document catalog (paginated). Returns doc IDs, summaries, and token counts.
   - offset: starting index (default 0)
   - limit: docs per page (default {list_docs_page_size})
   - Returns: {{"total_docs": int, "docs": [{{"doc_id", "summary", "tokens"}}], "has_more": bool}}
   - Use to browse more documents beyond the initial catalog.

5. read_lines(doc_id, start_line, end_line)
   - Read specific lines from a document (line numbers from grep/search results).
   - start_line: 1-based inclusive (default 1)
   - end_line: 1-based inclusive (default 50, max {read_lines_max_range})
   - Returns: {{"doc_id", "total_lines", "start_line", "end_line", "lines": "numbered text"}}
   - Use after grep/search to read more context around a match. Be generous — read ±20 lines around matches.

Strategy:
- FIRST: review the doc_catalog summaries to identify which documents might be relevant.
- ALWAYS start your first turn with BOTH a search() call AND a grep() call in parallel:
  - search(): use the key nouns/topics from the query as a natural language search query
  - grep(): use a specific name, term, or pattern from the query
  This dual approach ensures broad coverage — search catches semantic matches (synonyms, related terms) while grep catches exact text.
- After the first turn, use results to guide deeper investigation:
  - Use grep_doc() to thoroughly search within specific documents identified by search or grep
  - Use read_lines() to expand context around matches (read ±20 lines minimum)
- For aggregation queries ("how many", "total", "list all", "what is the order"):
  - You MUST find ALL mentions across ALL documents, not just the first match
  - After finding initial matches, search for MORE related items in OTHER documents
  - Use multiple grep patterns with different keywords/synonyms
  - Check the doc_catalog for any other documents that might contain related info
- For temporal reasoning ("how many days/weeks/months ago", "which happened first", "what order"):
  - Look for date/timestamp patterns like `time=YYYY-MM-DD` or `seq=N,time=YYYY-MM-DD` near relevant events
  - Compare timestamps carefully — the seq number and time fields indicate when events occurred
- For knowledge-update questions ("what is my current X", "where did Y move to"):
  - When multiple entries mention the same fact with different values, the LATEST entry (highest seq number or most recent timestamp) takes precedence
- You can call multiple tools in one turn (parallel execution).
- You have {max_iterations} turns total. Be thorough but efficient.

Output format (strict JSON, no other text):

To call tools:
{{
  "tool_calls": [
    {{"tool": "grep", "args": {{"pattern": "...", "limit": 30}}}},
    {{"tool": "search", "args": {{"query": "...", "limit": 5}}}},
    {{"tool": "read_lines", "args": {{"doc_id": "...", "start_line": 1, "end_line": 50}}}}
  ],
  "reasoning": "brief explanation of your strategy"
}}

To finish (when you have enough information or exhausted options):
{{
  "done": true,
  "relevant_snippets": [
    {{"doc_id": "...", "start_line": 5, "end_line": 40}},
    ...
  ],
  "relevant_docs": ["doc_id_1", "doc_id_2"],
  "reasoning": "brief summary of findings"
}}

Rules for finishing:
- relevant_snippets: list of (doc_id, start_line, end_line) tuples identifying line ranges containing relevant information. Include GENEROUS context — the full section around the relevant facts, not just the exact matching lines. Aim for 20-50 lines per snippet.
- relevant_docs: (optional) list of doc_ids where the ENTIRE document is relevant. Use this when most of a document is relevant to the query.
- Between relevant_snippets and relevant_docs, prefer relevant_snippets for precision. Use relevant_docs only when a large portion of the document is needed.
- If nothing relevant was found, set both to empty lists.
- Maximum {search_limit} items total (snippets + docs combined).

Output only valid JSON. No other text."""


__all__ = [
    "SYSTEM_DEFAULT",
    "EXTRACT_MEMORY_PROMPT",
    "REWRITE_CURRENT_PROMPT",
    "PLAN_UPDATE_PROMPT",
    "REWRITE_DOC_PROMPT",
    "SELECT_MERGE_GROUPS_PROMPT",
    "MERGE_DOCS_PROMPT",
    "SEARCH_MEMORY_PROMPT",
    "ANSWER_WITH_CONTEXT_PROMPT",
    "EVAL_JUDGE_PROMPT",
    "SUMM_FLUENCY_PROMPT",
    "SUMM_RECALL_PROMPT",
    "SUMM_PRECISION_PROMPT",
    "SUMM_EXTRACT_KEYPOINTS_PROMPT",
    "AGENTIC_QUERY_ANALYSIS_PROMPT",
    "AGENTIC_RELEVANCE_EVAL_PROMPT",
    "AGENTIC_TOOL_AGENT_PROMPT",
]
