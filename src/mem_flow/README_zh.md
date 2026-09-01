# mem_flow

[English](README.md) | **中文**

`mem_flow` 是 [Infini Memory](../../README_zh.md) 的底层记忆生命周期引擎。它将在线提取、定时维护与在线检索解耦，并始终以 Markdown 作为权威记忆源。

模块默认使用 S3 兼容对象存储，也可以配置为本地文件系统。每个实例创建时固定绑定一个记忆库和用户，后续操作都被限制在同一隔离作用域内。

## 核心特性

- **流程解耦**：提取、维护和检索可以独立部署与扩缩容。
- **可审计存储**：事实、来源、时间和主题索引以可读的 Markdown 与 JSON 保存。
- **缓冲式维护**：高频写入进入实例独占的 `CURRENT`，再由异步维护任务集中整理。
- **主题化记忆**：长期事实追加到不可变叶子文档，由自动生成的 `TOPIC.md` 提供导航。
- **混合检索**：支持分层、结构化、LLM、BM25、BM25 分区和 Agentic 等策略。
- **可插拔存储**：支持 AWS S3、MinIO、SeaweedFS 和本地目录。
- **可观测性**：内置结构化日志与 Prometheus 指标。

## 快速开始

### 安装

```bash
pip install infini-memory
```

需要 Python 3.13 或更高版本。Agentic 检索还需要安装：

```bash
pip install "infini-memory[deepagents]"
```

### 本地存储示例

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
    instance_id="api-worker-01",  # 可选；省略时自动生成
)

flow.extract(
    ExtractionRequest(
        messages=[ChatMessage(role="user", content="我喜欢无糖咖啡。")]
    )
)

# 对当前作用域定期运行一个维护任务。
flow.maintain(MaintenanceRequest())

result = flow.search(SearchRequest(query="我喜欢什么咖啡？"))
print(result.answer)
```

`ExtractionRequest.infer` 默认为 `True`，由 LLM 筛选值得长期保存的事实。只有输入是可信、明确且无需自动过滤的记忆时，才应设置 `infer=False`。

### S3 兼容存储

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

## 架构

```text
消息 ── 提取 ──> CURRENT ──┐
  └───────────> EVIDENCE   │
                            v
                   RAW -> REWRITE -> ROUTE
                                        │
                                        v
                            doc/<directory>/
                            ├── TOPIC.md
                            └── memory_<digest>.md

检索 <── evidence / current / raw / rewrite / doc / 可重建索引
```

### 提取

`MemoryExtractor` 保存来源证据，从输入中提取长期有效的原子事实，并追加到 `CURRENT_<instance_id>.md`。缓冲区超过限制后会轮转为带序号的完整文件。`instance_id` 在一个 `MemFlow` 实例的生命周期内保持不变。

### 维护

`MemoryMaintainer` 将 `CURRENT` 归档为 `RAW`，分批聚合重写，把每条事实路由到唯一的主题目录，追加新的不可变记忆叶子，刷新 `TOPIC.md`，最后清理已处理的中间文件。

同一 `(store_id, user_id)` 作用域同一时间只能运行一个维护任务。模块不提供分布式锁或事务。

### 检索

`MemoryRetriever` 同时检索活动记忆和长期记忆。默认 `HIERARCHICAL` 策略先选择工作文档和主题目录，再加载入选目录中的叶子文档。`AUTO` 会为时间、聚合、状态和偏好等问题增加结构化规划。

其他策略包括 `LLM`、`BM25`、`BM25_partition`、LLM/BM25 组合策略、`FOLDER_BM25_partition` 和 `AGENTIC`。可以通过 `SearchRequest.sources` 限定检索来源。

## 存储布局

```text
[<fixed_prefix>/]STORE_<store_id>/USER_<user_id>/
├── current/       # 实例独占的活动缓冲区
├── evidence/      # 不可变来源记录
├── raw/           # 等待维护的归档缓冲区
├── rewrite/       # 重写结果与持久化路由决策
├── index/v1/      # 可重建的结构化索引
└── doc/
    └── dir_<timestamp>_<id>/
        ├── TOPIC.md
        └── memory_<digest>.md
```

作用域存储适配器只暴露相对键，并拒绝路径穿越。`TOPIC.md` 是自动生成的导航索引，回答依据来源或记忆正文。结构化索引可以随时从 Markdown 重建：

```python
flow.rebuild_index()
flow.validate_index()
```

## 文档管理

长期记忆叶子可以按文档 ID 读取、按从 1 开始的闭区间行号修改，或直接删除：

```python
from mem_flow import DocLineUpdateRequest, DocReadRequest

doc = flow.get_doc(DocReadRequest(document_id="memory_..."))
flow.update_doc_lines(
    DocLineUpdateRequest(
        document_id=doc.document_id,
        start_line=3,
        end_line=3,
        replacement="- <seq=1785398400> 用户喜欢燕麦拿铁。",
    )
)
flow.delete_doc(DocReadRequest(document_id=doc.document_id))
```

这些接口只暴露文档正文，不暴露 Front Matter 或物理存储路径。`flow.delete_all()` 会删除当前绑定 store/user 作用域内的全部对象。

## 配置

主要默认值：

| 配置 | 默认值 | 用途 |
| --- | --- | --- |
| `storage.type` | `s3` | 选择 S3 兼容存储或本地存储 |
| `llm.model` | `deepseek-v4-flash-0731` | OpenAI 兼容模型名称 |
| `extraction.current_max_tokens` | `5000` | 活动缓冲区轮转阈值 |
| `maintenance.rewrite_batch_max_tokens` | `12000` | 单个重写批次上限 |
| `index.enabled` | `true` | 维护可重建的结构化索引 |
| `retrieval.strategy` | `HIERARCHICAL` | 默认检索策略 |
| `metrics.enabled` | `true` | 输出流程与依赖指标 |

所有经过校验的配置项见 [`config.py`](config.py)。

## 测试

```bash
uv sync
uv run pytest tests/mem_flow -v
```

真实 LLM 与 S3 测试需要显式开启：

```bash
uv run pytest tests/mem_flow/test_mem_flow.py -v --mem-flow-real-llm
uv run pytest tests/mem_flow/test_mem_flow.py -v --mem-flow-real-s3
```

## 许可证

[Apache License 2.0](../../LICENSE)
