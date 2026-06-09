from __future__ import annotations

import dataclasses
import datetime
import logging
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from zoneinfo import ZoneInfo

from . import term_color


class SearchStrategy(str, Enum):
    """Search strategy enum."""
    LLM = "LLM"
    BM25 = "BM25"
    LLM_AND_BM25 = "LLM_and_BM25"
    LLM_AND_BM25_PARTITION = "LLM_and_BM25_partition"
    LLM_OR_BM25 = "LLM_or_BM25"
    BM25_partition = "BM25_partition"
    FOLDER_BM25_PARTITION = "FOLDER_BM25_partition"
    AGENTIC = "AGENTIC"

try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore


# Color constants are now unified from term_color


@dataclass
class LogConfig:
    level: str = "INFO"


@dataclass
class LLMConfig:
    openai_api_key: str = ""
    openai_base_url: str = ""
    model: str = "gpt-5-mini"
    judge_model: str = "gpt-5-mini"  # Model used for evaluation judge stage
    temperature: float = 0.1  # LLM temperature parameter, controls output randomness
    retry_max_attempts: int = 10  # Max retry attempts for LLM calls
    retry_initial_wait: float = 5.0  # Initial wait time for LLM retries (seconds)
    retry_max_wait: float = 60.0  # Max wait time for LLM retries (seconds)
    retry_jitter: float = 0.2  # LLM retry wait jitter ratio (0-1)


@dataclass
class LangfuseConfig:
    public_key: str = ""
    secret_key: str = ""
    host: str = ""
    project: str = ""


@dataclass
class CommonConfig:
    env: str = "dev"
    timezone: str = "Asia/Shanghai"


@dataclass
class MemoryConfig:
    enabled: bool = False
    data_root: str = "data"
    doc_dir: str = "doc"
    metadata_dir: str = "metadata"
    index_file: str = "index.json"
    markdown_length: int = 5000  # Approximate token limit (split threshold for non-CURRENT documents)
    max_current_length: int = 5000  # Max token count for CURRENT document (split only after reaching this value)
    current_stale_seconds: int = 3600  # CURRENT doc staleness timeout (seconds). When the CURRENT doc has not been updated for this long, the next write triggers SPLIT/MERGE first. 0 = disabled.
    summary_length: int = 100    # Approximate token limit for summaries
    search_limit: int = 50       # Max number of documents returned by search
    search_strategy: str = SearchStrategy.LLM_AND_BM25_PARTITION  # Search strategy
    search_folder: str = ""      # Directory to search in FOLDER_BM25_partition strategy (e.g., raw, rewrite, doc)
    rewrite_doc_threads: int = 5  # Number of threads when calling REWRITE_DOC_PROMPT
    retry_max_attempts: int = 10  # Max retry attempts on add/search failure
    retry_initial_wait: int = 5  # Initial wait time on add/search failure (seconds)
    # Merge configuration
    merge_enabled: bool = True  # Whether to enable document merging
    merge_frequency: int = 10  # Trigger merge every N SPLIT calls
    merge_max_tokens: int = 1000  # Max token count for documents eligible for merging (only merge docs below this)
    merge_trigger_min_count: int = 50  # After SPLIT, also trigger MERGE if docs below merge_max_tokens exceed this count
    # PLAN_UPDATE_PROMPT split configuration
    plan_split_threshold: int = 5000  # Split content first if it exceeds this token count when calling PLAN_UPDATE_PROMPT
    plan_split_threads: int = 5  # Number of threads for parallel PLAN_UPDATE_PROMPT calls after preventive split
    # Agentic search configuration
    agentic_max_iterations: int = 7  # Max iteration rounds for agentic search
    agentic_min_relevant_docs: int = 1  # Min relevant docs for agentic search (can stop early once reached)
    agentic_grep_limit: int = 30  # Max matches returned by grep tool
    agentic_grep_context_lines: int = 3  # Default context lines for grep tool
    agentic_grep_max_line_length: int = 500  # Max characters per grep match line (truncated)
    agentic_read_lines_max_range: int = 100  # Max lines per read_lines call
    agentic_search_snippet_tokens: int = 200  # Max tokens for search result snippets
    agentic_list_docs_page_size: int = 20  # Documents per page for list_docs
    agentic_catalog_summary_length: int = 200  # Max characters for each document summary in initial doc catalog


class ColorFormatter(logging.Formatter):
    def __init__(self, fmt: str, datefmt: Optional[str] = None, *, enabled: bool = True):
        super().__init__(fmt, datefmt)
        self.enabled = enabled

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        level = record.levelno
        if getattr(record, "no_color", False):
            return msg
        if level >= logging.ERROR:
            return term_color.red(msg, enabled=self.enabled)
        if level >= logging.WARNING:
            return term_color.yellow(msg, enabled=self.enabled)
        if level == logging.DEBUG:
            return term_color.cyan(msg, enabled=self.enabled)
        return msg


class InfiniMemoryConfig:
    """Read configuration from `config/config.toml` and initialize logging."""

    def __init__(self, root: Optional[Path] = None, config_file: Optional[Path] = None):
        self.root = Path(root or Path.cwd())
        self.config_path = Path(config_file) if config_file else (self.root / "config" / "config.toml")
        self._enable_file_logging = True

        raw = self._load_toml(self.config_path)
        log_dict = (raw.get("log") or {}) if isinstance(raw, dict) else {}
        llm_dict = (raw.get("llm") or {}) if isinstance(raw, dict) else {}
        langfuse_dict = (raw.get("langfuse") or {}) if isinstance(raw, dict) else {}
        common_dict = (raw.get("common") or {}) if isinstance(raw, dict) else {}
        memory_dict = (raw.get("memory") or {}) if isinstance(raw, dict) else {}

        self.log = LogConfig(
            level=str(log_dict["level"]) if "level" in log_dict else LogConfig.level,
        )
        self.langfuse = LangfuseConfig(
            public_key=str(langfuse_dict["public_key"]) if "public_key" in langfuse_dict else str(os.getenv("LANGFUSE_PUBLIC_KEY", "")),
            secret_key=str(langfuse_dict["secret_key"]) if "secret_key" in langfuse_dict else str(os.getenv("LANGFUSE_SECRET_KEY", "")),
            host=str(langfuse_dict["host"]) if "host" in langfuse_dict else str(os.getenv("LANGFUSE_HOST", "")),
            project=str(langfuse_dict["project"]) if "project" in langfuse_dict else str(os.getenv("LANGFUSE_PROJECT", "")),
        )
        self.llm = LLMConfig(
            openai_api_key=str(llm_dict["openai_api_key"]) if "openai_api_key" in llm_dict else str(os.getenv("OPENAI_API_KEY", "")),
            openai_base_url=str(llm_dict["openai_base_url"]) if "openai_base_url" in llm_dict else str(os.getenv("OPENAI_BASE_URL", "")),
            model=str(llm_dict["model"]) if "model" in llm_dict else LLMConfig.model,
            judge_model=str(llm_dict["judge_model"]) if "judge_model" in llm_dict else LLMConfig.judge_model,
            temperature=float(llm_dict["temperature"]) if "temperature" in llm_dict else LLMConfig.temperature,
            retry_max_attempts=int(llm_dict["retry_max_attempts"]) if "retry_max_attempts" in llm_dict else LLMConfig.retry_max_attempts,
            retry_initial_wait=float(llm_dict["retry_initial_wait"]) if "retry_initial_wait" in llm_dict else LLMConfig.retry_initial_wait,
            retry_max_wait=float(llm_dict["retry_max_wait"]) if "retry_max_wait" in llm_dict else LLMConfig.retry_max_wait,
            retry_jitter=float(llm_dict["retry_jitter"]) if "retry_jitter" in llm_dict else LLMConfig.retry_jitter,
        )
        self.common = CommonConfig(
            env=str(common_dict["env"]) if "env" in common_dict else CommonConfig.env,
            timezone=str(common_dict["timezone"]) if "timezone" in common_dict else CommonConfig.timezone,
        )
        self.memory = MemoryConfig(
            enabled=bool(memory_dict["enabled"]) if "enabled" in memory_dict else MemoryConfig.enabled,
            data_root=str(memory_dict["data_root"]) if "data_root" in memory_dict else MemoryConfig.data_root,
            doc_dir=str(memory_dict["doc_dir"]) if "doc_dir" in memory_dict else MemoryConfig.doc_dir,
            metadata_dir=str(memory_dict["metadata_dir"]) if "metadata_dir" in memory_dict else MemoryConfig.metadata_dir,
            index_file=str(memory_dict["index_file"]) if "index_file" in memory_dict else MemoryConfig.index_file,
            markdown_length=int(memory_dict["markdown_length"]) if "markdown_length" in memory_dict else MemoryConfig.markdown_length,
            max_current_length=int(memory_dict["max_current_length"]) if "max_current_length" in memory_dict else MemoryConfig.max_current_length,
            current_stale_seconds=int(memory_dict["current_stale_seconds"]) if "current_stale_seconds" in memory_dict else MemoryConfig.current_stale_seconds,
            summary_length=int(memory_dict["summary_length"]) if "summary_length" in memory_dict else MemoryConfig.summary_length,
            search_limit=int(memory_dict["search_limit"]) if "search_limit" in memory_dict else MemoryConfig.search_limit,
            search_strategy=str(memory_dict["search_strategy"]) if "search_strategy" in memory_dict else MemoryConfig.search_strategy,
            retry_max_attempts=int(memory_dict["retry_max_attempts"]) if "retry_max_attempts" in memory_dict else MemoryConfig.retry_max_attempts,
            retry_initial_wait=int(memory_dict["retry_initial_wait"]) if "retry_initial_wait" in memory_dict else MemoryConfig.retry_initial_wait,
            plan_split_threshold=int(memory_dict["plan_split_threshold"]) if "plan_split_threshold" in memory_dict else MemoryConfig.plan_split_threshold,
            plan_split_threads=int(memory_dict["plan_split_threads"]) if "plan_split_threads" in memory_dict else MemoryConfig.plan_split_threads,
            agentic_max_iterations=int(memory_dict["agentic_max_iterations"]) if "agentic_max_iterations" in memory_dict else MemoryConfig.agentic_max_iterations,
            agentic_min_relevant_docs=int(memory_dict["agentic_min_relevant_docs"]) if "agentic_min_relevant_docs" in memory_dict else MemoryConfig.agentic_min_relevant_docs,
            agentic_grep_limit=int(memory_dict["agentic_grep_limit"]) if "agentic_grep_limit" in memory_dict else MemoryConfig.agentic_grep_limit,
            agentic_grep_context_lines=int(memory_dict["agentic_grep_context_lines"]) if "agentic_grep_context_lines" in memory_dict else MemoryConfig.agentic_grep_context_lines,
            agentic_grep_max_line_length=int(memory_dict["agentic_grep_max_line_length"]) if "agentic_grep_max_line_length" in memory_dict else MemoryConfig.agentic_grep_max_line_length,
            agentic_read_lines_max_range=int(memory_dict["agentic_read_lines_max_range"]) if "agentic_read_lines_max_range" in memory_dict else MemoryConfig.agentic_read_lines_max_range,
            agentic_search_snippet_tokens=int(memory_dict["agentic_search_snippet_tokens"]) if "agentic_search_snippet_tokens" in memory_dict else MemoryConfig.agentic_search_snippet_tokens,
            agentic_list_docs_page_size=int(memory_dict["agentic_list_docs_page_size"]) if "agentic_list_docs_page_size" in memory_dict else MemoryConfig.agentic_list_docs_page_size,
            agentic_catalog_summary_length=int(memory_dict["agentic_catalog_summary_length"]) if "agentic_catalog_summary_length" in memory_dict else MemoryConfig.agentic_catalog_summary_length,
        )

        self._setup_logging()

    def _load_toml(self, path: Path) -> dict:
        if tomllib is None:
            raise RuntimeError("Python 3.11+ requires built-in tomllib for TOML parsing")
        if not path.exists():
            return {}
        with path.open("rb") as f:
            return tomllib.load(f)

    @classmethod
    def from_kwargs(
        cls,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-5-mini",
        enabled: bool = True,
        data_root: str = "data",
        log_level: str = "WARNING",
        enable_file_logging: bool = False,
        root: Optional[Path] = None,
        **memory_kwargs: Any,
    ) -> "InfiniMemoryConfig":
        """Build a config programmatically without a TOML file.

        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            base_url: OpenAI base URL. Falls back to OPENAI_BASE_URL env var.
            model: LLM model name.
            enabled: Whether memory is enabled (default True).
            data_root: Root directory for document storage.
            log_level: Logging level (default WARNING for library use).
            enable_file_logging: Whether to create log files on disk (default False).
            root: Project root directory (default cwd).
            **memory_kwargs: Extra fields passed to MemoryConfig (e.g. search_strategy,
                markdown_length, search_limit).
        """
        instance = cls.__new__(cls)
        instance.root = Path(root or Path.cwd())
        instance.config_path = None
        instance._enable_file_logging = enable_file_logging

        instance.log = LogConfig(level=log_level)
        instance.langfuse = LangfuseConfig(
            public_key=str(os.getenv("LANGFUSE_PUBLIC_KEY", "")),
            secret_key=str(os.getenv("LANGFUSE_SECRET_KEY", "")),
            host=str(os.getenv("LANGFUSE_HOST", "")),
            project=str(os.getenv("LANGFUSE_PROJECT", "")),
        )
        instance.llm = LLMConfig(
            openai_api_key=api_key or str(os.getenv("OPENAI_API_KEY", "")),
            openai_base_url=base_url or str(os.getenv("OPENAI_BASE_URL", "")),
            model=model,
        )
        instance.common = CommonConfig()

        mem_fields = {f.name for f in dataclasses.fields(MemoryConfig)}
        mem_overrides = {k: v for k, v in memory_kwargs.items() if k in mem_fields}
        instance.memory = MemoryConfig(enabled=enabled, data_root=data_root, **mem_overrides)

        instance._setup_logging()
        return instance

    def _setup_logging(self) -> None:
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        level = level_map.get(self.log.level.upper(), logging.INFO)

        logger = logging.getLogger("infini_memory")
        logger.setLevel(level)
        logger.handlers.clear()

        # In pytest environment, avoid creating console handler; let pytest's log_cli handle output
        is_pytest = (
            os.getenv("PYTEST_CURRENT_TEST") is not None
            or "pytest" in sys.modules
        )
        # Control log propagation: propagate in pytest environment for log_cli to capture;
        # do not propagate in non-pytest environment, use custom console handler instead.
        logger.propagate = True if is_pytest else False

        # In pytest, ensure root logger level is not higher than desired level (e.g., DEBUG),
        # to prevent project DEBUG logs from being filtered by root logger.
        if is_pytest:
            root_logger = logging.getLogger()
            try:
                root_logger.setLevel(level)
            except Exception:
                pass

        # Format
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"

        # Console Handler (with color): only enabled in non-pytest environment; pytest uses its built-in handler
        if not is_pytest:
            console_handler = logging.StreamHandler()
            # Always use colored output
            console_handler.setFormatter(ColorFormatter(fmt, datefmt, enabled=True))
            console_handler.setLevel(level)
            logger.addHandler(console_handler)

        # File Handler (without color)
        if self._enable_file_logging:
            # Write to test_logs/ in pytest, otherwise write to logs/infini-memory/
            logs_dir = (
                self.root / "test_logs" if os.getenv("PYTEST_CURRENT_TEST") or "pytest" in sys.modules else self.root / "logs" / "infini-memory"
            )
            logs_dir.mkdir(parents=True, exist_ok=True)
            # Localized time (default Asia/Shanghai or configured timezone), log filename format: YYYY-MM-DD__HH-MM-SS.log
            now = datetime.datetime.now(datetime.UTC).astimezone(ZoneInfo(self.common.timezone))
            filename = now.strftime("%Y-%m-%d__%H-%M-%S") + ".log"
            file_path = logs_dir / filename

            file_handler = logging.FileHandler(file_path, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(fmt, datefmt))
            file_handler.setLevel(level)
            logger.addHandler(file_handler)

            # Create/update symlink to latest log file (production only, not pytest)
            symlink_path = logs_dir / "infini-memory.log"
            if not is_pytest:
                try:
                    # Remove existing symlink if present
                    if symlink_path.exists() or symlink_path.is_symlink():
                        symlink_path.unlink()
                    # Create new symlink
                    symlink_path.symlink_to(filename)
                    logger.debug("Log symlink updated: %s -> %s", symlink_path, filename)
                except Exception as e:
                    logger.debug("Failed to create log symlink: %s", e)

            # Example: output current log level
            logger.debug("Logging system initialized (DEBUG visible)")
            logger.info("Logging system initialized. level=%s, file=%s", self.log.level, file_path)

        # Limit third-party noisy logger levels to avoid DEBUG-level flooding
        # Only this project's namespace uses DEBUG; third-party loggers stay at INFO or higher
        noisy_loggers = [
            "openai",
            "openai._base_client",
            "httpcore",
            "httpx",
        ]
        for name in noisy_loggers:
            try:
                nl = logging.getLogger(name)
                # Respect externally set higher levels; otherwise raise DEBUG to INFO
                if nl.level == logging.NOTSET or nl.level < logging.INFO:
                    nl.setLevel(logging.INFO)
            except Exception:
                pass

    def rewrite_config(self, rewrite_items: List[str]) -> Tuple[int, List[Tuple[str, Any]]]:
        """Override configuration items based on command-line arguments.

        Args:
            rewrite_items: List of config overrides in "section.field:value" format,
                e.g.: ["memory.search_limit:5", "llm.temperature:0.2"]

        Returns:
            (error_count, list of successfully applied configs [(section.field, value), ...])

        Example:
            cfg = InfiniMemoryConfig()
            errors, applied = cfg.rewrite_config([
                "memory.search_limit:5",
                "llm.temperature:0.2",
                "memory.merge_enabled:true"
            ])
        """
        errors = 0
        applied: List[Tuple[str, Any]] = []

        # Mapping from section name to config object
        section_map = {
            "log": self.log,
            "llm": self.llm,
            "langfuse": self.langfuse,
            "common": self.common,
            "memory": self.memory,
        }

        for rewrite_item in rewrite_items:
            if ":" not in rewrite_item:
                logging.error("--rewrite-config format error, expected section.field:value (e.g., memory.search_limit:5)")
                errors += 1
                continue

            key, value_str = rewrite_item.split(":", 1)
            key = key.strip()
            value_str = value_str.strip()

            # Parse section and field
            if "." not in key:
                logging.error("--rewrite-config format error, expected section.field:value (missing section), skipping: %s", key)
                errors += 1
                continue

            section_name, field_name = key.split(".", 1)
            section_name = section_name.strip()
            field_name = field_name.strip()

            # Check if section exists
            if section_name not in section_map:
                logging.warning("Unknown config section: %s (skipping)", section_name)
                errors += 1
                continue

            section_obj = section_map[section_name]

            # Check if field exists in the config class
            if not hasattr(section_obj, field_name):
                logging.warning("Field %s does not exist in config section %s (skipping)", field_name, section_name)
                errors += 1
                continue

            # Get field type and current value
            field = dataclasses.fields(type(section_obj))[
                [f.name for f in dataclasses.fields(type(section_obj))].index(field_name)
            ]
            field_type = field.type

            # Type conversion
            converted_value = self._convert_value(value_str, field_type, section_name, field_name)
            if converted_value is None:
                errors += 1
                continue

            # Apply configuration
            setattr(section_obj, field_name, converted_value)
            applied.append((key, converted_value))
            logging.info("[rewrite-config] %s = %s", key, converted_value)

        return errors, applied

    def _convert_value(self, value_str: str, field_type: Type, section: str, field: str) -> Any:
        """Convert a string value to the target type.

        Args:
            value_str: String value to convert.
            field_type: Target type.
            section: Config section name (for error messages).
            field: Field name (for error messages).

        Returns:
            Converted value, or None if conversion fails.
        """
        # Handle Optional types
        if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
            # Get the actual type from Optional
            args = field_type.__args__
            for arg in args:
                if arg is not type(None):
                    field_type = arg
                    break

        # Normalize type (handle string-form type annotations)
        type_str = str(field_type).strip("'\"")

        # String type
        if field_type is str or field_type == str or type_str == "str":
            return value_str

        # Integer type
        if field_type is int or field_type == int or type_str == "int":
            try:
                return int(value_str)
            except ValueError:
                logging.error("Config %s.%s value should be an integer: %s", section, field, value_str)
                return None

        # Float type
        if field_type is float or field_type == float or type_str == "float":
            try:
                return float(value_str)
            except ValueError:
                logging.error("Config %s.%s value should be a float: %s", section, field, value_str)
                return None

        # Boolean type
        if field_type is bool or field_type == bool or type_str == "bool":
            lower_val = value_str.lower()
            if lower_val in ("true", "1", "yes", "y"):
                return True
            elif lower_val in ("false", "0", "no", "n"):
                return False
            else:
                logging.error("Config %s.%s value should be a boolean (true/false/1/0/yes/no): %s", section, field, value_str)
                return None

        # Enum type
        if hasattr(field_type, "__mro__") and Enum in field_type.__mro__:
            for member in field_type:
                if member.value == value_str or member.name == value_str:
                    return member
            logging.error("Config %s.%s value should be one of the enum values: %s (valid values: %s)",
                         section, field, value_str,
                         [m.value for m in field_type])
            return None

        # Unsupported type
        logging.error("Config %s.%s type %s does not support automatic conversion", section, field, field_type)
        return None


__all__ = [
    "InfiniMemoryConfig",
    "LogConfig",
    "LLMConfig",
    "LangfuseConfig",
    "CommonConfig",
    "MemoryConfig",
    "SearchStrategy",
]
