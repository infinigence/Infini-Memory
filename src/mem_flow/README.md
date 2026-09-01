# mem_flow

**English** | [中文](README_zh.md)

`mem_flow` is the lower-level lifecycle engine of [Infini Memory](../../README.md). It separates online extraction, scheduled maintenance, and online retrieval while keeping Markdown as the authoritative memory source.

It supports S3-compatible object storage by default and local filesystem storage when configured. Each instance is permanently bound to one store and one user, so all operations remain inside the same isolated scope.

## Key Features

- **Independent flows**: extraction, maintenance, and retrieval can be deployed and scaled separately.
- **Auditable storage**: facts, provenance, timestamps, and topic indexes are stored as readable Markdown and JSON.
- **Buffered maintenance**: high-frequency writes go to per-instance `CURRENT` files and are consolidated asynchronously.
- **Topic-oriented memory**: maintained facts are appended to immutable leaf documents, with generated `TOPIC.md` files for navigation.
- **Hybrid retrieval**: hierarchical, structured, LLM, BM25, partitioned BM25, and Agentic strategies are available.
- **Pluggable storage**: use AWS S3, MinIO, SeaweedFS, or a local directory.
- **Observability**: structured logs and Prometheus metrics are built in.

## Quick Start

### Installation

```bash
pip install infini-memory
```

Python 3.13 or later is required. The Agentic retrieval strategy additionally requires:

```bash
pip install "infini-memory[deepagents]"
```

### Local Example

```python
from mem_flow import (
    ChatMessage,
    ExtractionRequest,
    LLMConfig,
    MaintenanceRequest,
    MemFlow,
    MemFlowConfig,
    SearchRequest,
    StorageConfig,
)

flow = MemFlow.create(
    MemFlowConfig(
        storage=StorageConfig(type="local", path="data/mem_flow"),
        llm=LLMConfig(
            api_key="sk-...",
            base_url="https://api.openai.com/v1",
            model="gpt-5-mini",
        ),
    ),
    store_id="assistant",
    user_id="alice",
    instance_id="api-worker-01",  # Optional; generated when omitted
)

flow.extract(
    ExtractionRequest(
        messages=[ChatMessage(role="user", content="I prefer sugar-free coffee.")]
    )
)

# Run this periodically in a single maintenance job for the bound scope.
flow.maintain(MaintenanceRequest())

result = flow.search(SearchRequest(query="What kind of coffee do I prefer?"))
print(result.answer)
```

`ExtractionRequest.infer` defaults to `True`, which uses the LLM to select durable facts. Set `infer=False` only for trusted, explicit memory that should be appended without automatic filtering.

### S3-Compatible Storage

```python
from mem_flow import LLMConfig, MemFlow, MemFlowConfig, S3Config

flow = MemFlow.create(
    MemFlowConfig(
        s3=S3Config(
            endpoint_url="http://minio:9000",
            bucket="infini-memory",
            access_key="access-key",
            secret_key="secret-key",
            fixed_prefix="inf_mem",
        ),
        llm=LLMConfig(api_key="sk-...", model="gpt-5-mini"),
    ),
    store_id="assistant",
    user_id="alice",
)
```

## Architecture

```text
Messages ── Extract ──> CURRENT ──┐
    └───────────────> EVIDENCE    │
                                  v
                         RAW -> REWRITE -> ROUTE
                                              │
                                              v
                                  doc/<directory>/
                                  ├── TOPIC.md
                                  └── memory_<digest>.md

Search <── evidence / current / raw / rewrite / doc / rebuildable index
```

### Extract

`MemoryExtractor` records source evidence, extracts durable atomic facts, and appends them to `CURRENT_<instance_id>.md`. Oversized buffers rotate to numbered full files. An `instance_id` is reused for the lifetime of a `MemFlow` instance.

### Maintain

`MemoryMaintainer` archives `CURRENT` files to `RAW`, batches and rewrites facts, routes every fact to exactly one topic directory, appends a new immutable memory leaf, refreshes `TOPIC.md`, and then removes processed intermediate files.

Run only one maintenance task at a time for a `(store_id, user_id)` scope. The module does not provide distributed locks or transactions.

### Retrieve

`MemoryRetriever` searches active and maintained memory. The default `HIERARCHICAL` strategy first selects working documents and topic directories, then loads leaf documents from selected directories. `AUTO` adds structured planning for temporal, aggregation, state, and preference queries.

Other strategies include `LLM`, `BM25`, `BM25_partition`, combined LLM/BM25 variants, `FOLDER_BM25_partition`, and `AGENTIC`. Results can be limited to selected sources with `SearchRequest.sources`.

## Storage Layout

```text
[<fixed_prefix>/]STORE_<store_id>/USER_<user_id>/
├── current/       # Active per-instance buffers
├── evidence/      # Immutable source records
├── raw/           # Archived buffers awaiting maintenance
├── rewrite/       # Rewrites and persisted routing decisions
├── index/v1/      # Rebuildable structured sidecar
└── doc/
    └── dir_<timestamp>_<id>/
        ├── TOPIC.md
        └── memory_<digest>.md
```

The scoped storage adapter exposes only relative keys and rejects path traversal. `TOPIC.md` is a generated navigation index; answers are grounded in source or memory documents. The structured index is disposable and can be rebuilt from Markdown:

```python
flow.rebuild_index()
flow.validate_index()
```

## Document Management

Maintained memory leaves can be read, edited by inclusive one-based line ranges, or deleted by document ID:

```python
from mem_flow import DocLineUpdateRequest, DocReadRequest

doc = flow.get_doc(DocReadRequest(document_id="memory_..."))
flow.update_doc_lines(
    DocLineUpdateRequest(
        document_id=doc.document_id,
        start_line=3,
        end_line=3,
        replacement="- <seq=1785398400> The user prefers oat milk latte.",
    )
)
flow.delete_doc(DocReadRequest(document_id=doc.document_id))
```

These APIs expose only document bodies, not front matter or physical storage paths. `flow.delete_all()` removes every object in the bound store/user scope.

## Configuration

Important defaults:

| Setting | Default | Purpose |
| --- | --- | --- |
| `storage.type` | `s3` | Select S3-compatible or local storage |
| `llm.model` | `deepseek-v4-flash-0731` | OpenAI-compatible model name |
| `extraction.current_max_tokens` | `5000` | Rotate the active buffer |
| `maintenance.rewrite_batch_max_tokens` | `12000` | Maximum rewrite batch size |
| `index.enabled` | `true` | Maintain the rebuildable structured index |
| `retrieval.strategy` | `HIERARCHICAL` | Default retrieval strategy |
| `metrics.enabled` | `true` | Export flow and dependency metrics |

See [`config.py`](config.py) for all validated options.

## Testing

```bash
uv sync
uv run pytest tests/mem_flow -v
```

Real LLM and S3 checks are opt-in:

```bash
uv run pytest tests/mem_flow/test_mem_flow.py -v --mem-flow-real-llm
uv run pytest tests/mem_flow/test_mem_flow.py -v --mem-flow-real-s3
```

## License

[Apache License 2.0](../../LICENSE)
