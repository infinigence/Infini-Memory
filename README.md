# Infini Memory

[![PyPI](https://img.shields.io/pypi/v/infini-memory)](https://pypi.org/project/infini-memory/)
[![Python](https://img.shields.io/pypi/pyversions/infini-memory)](https://pypi.org/project/infini-memory/)
[![License](https://img.shields.io/github/license/infinigence/Infini-Memory)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2606.10677-b31b1b)](https://arxiv.org/abs/2606.10677)

**English** | [中文](README_zh.md)

A maintainable, text-based persistent memory architecture that organizes LLM agent memory as topic-structured documents.

[[Paper](https://arxiv.org/abs/2606.10677)]

## Introduction

Long-term LLM agents need persistent memory that can track changing facts and provide relevant evidence across sessions. Existing memory systems often store observations as isolated records, summaries, or indexed fragments, which leads to four recurring problems:

- **Memory fragmentation**: Evidence about the same user, task, or event is scattered across many small records
- **Memory conflict**: Newer observations contradict older ones, but append-style storage keeps both versions retrievable
- **Compression loss**: Summarization weakens temporal order and source cues
- **Insufficient retrieval**: Single-shot retrieval returns isolated fragments rather than enough evidence for multi-hop reasoning

Infini Memory addresses these by treating persistent memory as a **lifecycle maintenance problem** with three coupled operations — **write**, **maintain**, and **read**. It represents memory as topic-structured documents, where each document groups related evidence under a shared subject and carries entry-level metadata (sequence numbers, timestamps, source tags) that retains temporal and provenance cues as content is rewritten.

### Key Design Choices

- **Topic documents as memory carrier**: Plain-text Markdown documents organized by topic, no dependency on vector or graph databases
- **Buffered writing with periodic consolidation**: High-frequency writes are appended to a `CURRENT` buffer; consolidation (rewriting, splitting, updating, merging) is triggered when enough information accumulates or a time threshold is reached
- **Agentic retrieval**: The LLM iteratively searches, verifies, and expands evidence through memory tools, rather than relying on a single retrieval step
- **File-system backend**: No external infrastructure required — the memory state remains readable, editable, and portable

## Quick Start

### Installation

```bash
pip install infini-memory
```

Infini Memory requires an LLM to function, with `gpt-5-mini` from OpenAI as the default. Set your API key:

```bash
export OPENAI_API_KEY="sk-..."
```

### chat_with_memory example

A runnable version is available at [`examples/chat_with_memory.py`](examples/chat_with_memory.py).

Sample runs:

```text
uv run python examples/chat_with_memory.py
Chat with AI (type 'exit' to quit)

You: What is my name?
AI: I don't know your name unless you tell me. Would you like to share it now? You can also just tell me what to call you in this conversation, or whether you'd like me to remember it (no problem if you'd rather not — I can simply call you "you" or anything else).

You: My name is Jay
AI: Got it, Jay — I'll call you that for the rest of this conversation. Would you like me to remember the name so I can recognize you next time, or do you prefer something else? You can change it any time.

You: exit
Goodbye!
```

```text
uv run python examples/chat_with_memory.py
Chat with AI (type 'exit' to quit)

You: What is my name?
AI: Your name is Jay. I'll address you that way; let me know any time if you want to change it or have me forget it.

You: What food do I like?
AI: I don't know what you like to eat yet — so far my memory only has your name, Jay. Tell me your taste preferences and I can give better suggestions, or remember them for next time (with your permission).

You: I like blueberries and strawberries
AI: Great, Jay! Blueberries and strawberries — a lovely combo. Want me to save "likes blueberries and strawberries" to your preferences so I can use it in future recommendations?

You: exit
Goodbye!
```

```text
uv run python examples/chat_with_memory.py
Chat with AI (type 'exit' to quit)

You: My name and preferences
AI: Here's what I have on record for you:
- Name: Jay
- Preferences: likes blueberries and strawberries (saved)

Want me to change or add anything? Should I prioritize these preferences when suggesting recipes or orders going forward?
```

> **Note:** `chat_with_memory.py` is only a demonstration of the memory effect — it extracts memory on every turn. In production, you can batch turns together and extract memory once every N rounds to reduce LLM calls.

## Document Management

The `Memory` class provides full CRUD operations for documents and user management:

```python
from infini_memory import Memory

memory = Memory()

# List all documents for a user (metadata only)
docs = memory.list(user_id="alice")

# Get all documents with content
all_docs = memory.get_all(user_id="alice")

# Get a specific document by ID
doc = memory.get("2025-01-15_14-30-00_abc123", user_id="alice")
print(doc["content"], doc["summary"])

# Update a document
memory.update("2025-01-15_14-30-00_abc123", "new content", "new summary", user_id="alice")

# Delete a document
memory.delete("2025-01-15_14-30-00_abc123", user_id="alice")

# Document count and statistics
n = memory.count(user_id="alice")
stats = memory.stats(user_id="alice")  # {"total_docs", "avg_tokens", ...}

# Operation history
events = memory.history(user_id="alice")

# Delete all documents (keeps user directory)
memory.delete_all(user_id="alice")

# List all users
users = memory.list_users()  # ["alice", "bob", ...]

# Delete all data for a user
memory.delete_user("alice")

# Reset: delete all data for all users
memory.reset()
```

## Configuration

### Programmatic (recommended for library use)

```python
from infini_memory import Memory

memory = Memory(
    api_key="sk-...",                      # or set OPENAI_API_KEY env var
    base_url="https://api.openai.com/v1",  # custom endpoint
    model="gpt-5-mini",                   # LLM model
    data_root="my_memory_data",            # where documents are stored
    search_strategy="AGENTIC",             # retrieval strategy
    markdown_length=2000,                  # document token limit before splitting
)
```

### TOML config file (for standalone deployment)

```python
from pathlib import Path
from infini_memory import InfiniMemory, InfiniMemoryConfig

cfg = InfiniMemoryConfig(config_file=Path("config/config.toml"))
mem = InfiniMemory()

mem.add(messages, user_id="user_001", cfg=cfg)
result = mem.search("query", user_id="user_001", cfg=cfg)
```

Key fields in `config.toml`:

```toml
[llm]
openai_api_key = "sk-..."
model = "gpt-5-mini"

[memory]
enabled = true
data_root = "data"
markdown_length = 1000
search_strategy = "AGENTIC"
```

## Architecture

### Topic Document Format

Infini Memory stores persistent memory as topic documents, where each document groups related facts, preferences, and event cues under a shared topic. A document contains a metadata header (`id`, `summary`, `token_count`, `created_time`, `update_log`, `aux`) and a hierarchical body. The body uses topic and subtopic headings to organize memory entries, each prefixed with a parsable signature `<seq=..., time=..., source=...>` that preserves temporal order, provenance, and revision context.

<p align="center">
  <img src="images/Memory_Document_Format.svg" width="500" alt="Topic Document Format">
</p>

### Write Path

The writing and consolidation pipeline separates high-frequency writes from low-frequency structural maintenance. New memories are first appended to a `CURRENT` buffer, then periodically consolidated into the topic document library.

<p align="center">
  <img src="images/Memory_Extraction_Consolidation.svg" width="500" alt="Memory Writing and Consolidation Pipeline">
</p>

1. **Extract**: LLM extracts salient information from conversations into structured Markdown
2. **Append**: New content is appended to the `CURRENT` buffer document
3. **Rewrite**: When the buffer exceeds a token threshold or a time window, content is aggregated by topic into `REWRITE_CURRENT`
4. **Split & Update**: A planner routes rewritten content into topic documents (new or existing), reconciling contradictions and preserving metadata
5. **Merge**: Small documents on similar topics are periodically merged; summaries and metadata are refreshed

### Read Path

Infini Memory supports two retrieval variants.

**Hybrid Retrieval (LLM Summary + BM25 Partitions):** The LLM selects candidate documents by summary relevance, and BM25 supplements with lexically matched partitions from the remaining documents.

<p align="center">
  <img src="images/Memory_Retrieval_BM25.svg" width="500" alt="Hybrid Retrieval: LLM Summary + BM25 Partitions">
</p>

**Agentic Retrieval:** The LLM agent iteratively calls memory tools (`grep`, `grep_doc`, `search`, `list_docs`, `read_lines`) to search, verify, and expand evidence across topic documents and the `CURRENT` buffer before generating the final answer. When the agent returns insufficient evidence, BM25-based partition retrieval supplements the results.

<p align="center">
  <img src="images/Memory_Retrieval_Agentic.svg" width="500" alt="Agentic Retrieval">
</p>

## Citation

```bibtex
@misc{ji2026infinimemorymaintainabletopic,
      title={Infini Memory: Maintainable Topic Documents for Long-Term LLM Agent Memory}, 
      author={Suozhao Ji and Baodong Wu and Zehao Wang and Lei Xia and Qingping Li and Ruisong Wang and Wenbo Ding and Zhenhua Zhu and Boxun Li and Guohao Dai and Yu Wang},
      year={2026},
      eprint={2606.10677},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2606.10677}, 
}
```

## License

[Apache License 2.0](LICENSE)
