"""Chat with an AI assistant that remembers your conversations.

Prerequisites:
    pip install infini-memory openai

    export OPENAI_API_KEY="sk-..."
    export OPENAI_BASE_URL="https://api.openai.com/v1"  # optional
    export OPENAI_MODEL="gpt-5-mini"                    # optional

Optional local-memory settings:
    export INFINI_MEMORY_DATA_ROOT="data"
    export INFINI_MEMORY_STORE="default"
    export INFINI_MEMORY_USER_ID="default_user"

Usage:
    python examples/chat_with_memory.py
"""

import atexit
import logging
import os
import readline  # noqa: F401  # enable line editing (arrow keys, delete) in input()
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from infini_memory_classic import Memory

# -- ANSI escape codes -------------------------------------------------------
BLUE_BOLD = "\033[1;34m"
ORANGE_BOLD = "\033[1;38;5;208m"
RESET = "\033[0m"

# -- Configuration ------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
_DATA_PATH = Path(os.getenv("INFINI_MEMORY_DATA_ROOT", "data")).expanduser()
if not _DATA_PATH.is_absolute():
    _DATA_PATH = _PROJECT_ROOT / _DATA_PATH
_DATA_PATH = _DATA_PATH.resolve()
_STORE = os.getenv("INFINI_MEMORY_STORE", "default")
_USER_ID = os.getenv("INFINI_MEMORY_USER_ID", "default_user")

# This example intentionally uses local storage. LLM settings come from the
# environment, so no project config file is required.
memory = Memory(
    model=_MODEL,
    data_root=_DATA_PATH.name,
    root=_DATA_PATH.parent,
    storage_type="local",
    search_strategy="BM25_partition",
)
openai_client = OpenAI()

# Suppress infini_memory_classic console logs so they don't interleave with chat I/O.
# File logging is unaffected.
_im_logger = logging.getLogger("infini_memory_classic")
for _h in _im_logger.handlers:
    if isinstance(_h, logging.StreamHandler) and not isinstance(
        _h, logging.FileHandler
    ):
        _h.setLevel(logging.WARNING)

# Background pool for memory.add() so the chat loop doesn't block on LLM extraction.
_memory_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="memory-add")

BASE_SYSTEM_PROMPT = "You are a helpful AI. Answer the question based on query and memories."

# Persistent counter file for the memory `seq` parameter. Survives restarts so
# each turn across sessions gets a unique, monotonically increasing seq.
_SEQ_FILE = Path(__file__).parent / "demo_info" / "seq.txt"


def _load_seq() -> int:
    try:
        return int(_SEQ_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return 0


def _save_seq(value: int) -> None:
    _SEQ_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SEQ_FILE.write_text(str(value))


_seq_counter = _load_seq()

# Persistent in-process conversation history (user/assistant turns only).
conversation_history: list[dict] = []


def _add_memory_async(messages: list, store: str, user_id: str, seq: int) -> None:
    def _run() -> None:
        try:
            memory.add(messages, store=store, user_id=user_id, seq=seq)
        except Exception as e:
            print(f"\n[memory.add failed in background] {e}")

    _memory_executor.submit(_run)


def _shutdown_executor() -> None:
    _memory_executor.shutdown(wait=True)


atexit.register(_shutdown_executor)


def _format_memories(relevant_memories) -> str:
    if not relevant_memories or not isinstance(relevant_memories, dict):
        return ""
    lines = []
    for entry in relevant_memories.get("results", []):
        text = (entry.get("summary") or "").strip()
        if not text:
            text = (entry.get("content") or "").strip()
        if text:
            lines.append(f"- {text}")
    return "\n".join(lines)


def _build_memory_system_message(
    message: str, store: str, user_id: str
) -> dict | None:
    relevant_memories = memory.search(query=message, store=store, user_id=user_id)
    memories_str = _format_memories(relevant_memories)
    if not memories_str:
        return None
    return {
        "role": "system",
        "content": (
            "Consider the following user memories when answering.\n"
            f"User Memories:\n{memories_str}"
        ),
    }


def chat_with_memories(
    message: str, user_id: str = _USER_ID, store: str = _STORE
) -> str:
    global _seq_counter
    _seq_counter += 1
    seq = _seq_counter
    _save_seq(seq)

    user_msg = {"role": "user", "content": message}

    request_messages: list[dict] = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]
    memory_sys_msg = _build_memory_system_message(message, store, user_id)
    if memory_sys_msg is not None:
        request_messages.append(memory_sys_msg)
    request_messages.extend(conversation_history)
    request_messages.append(user_msg)

    stream = openai_client.chat.completions.create(
        model=_MODEL, messages=request_messages, stream=True
    )
    sys.stdout.write(f"{ORANGE_BOLD}【AI】：{RESET}")
    sys.stdout.flush()
    chunks = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            chunks.append(delta)
            sys.stdout.write(f"{ORANGE_BOLD}{delta}{RESET}")
            sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    assistant_response = "".join(chunks)
    assistant_msg = {"role": "assistant", "content": assistant_response}

    conversation_history.append(user_msg)
    conversation_history.append(assistant_msg)
    _add_memory_async([user_msg, assistant_msg], store, user_id, seq)

    return assistant_response


def main():
    print("Chat with AI (type 'exit' to quit)\n")
    try:
        while True:
            try:
                user_input = input(f"{BLUE_BOLD}【You】：{RESET}").strip()
            except EOFError:
                break
            if not user_input:
                continue
            if user_input.lower() == "exit":
                break
            chat_with_memories(user_input)
            print()
    except KeyboardInterrupt:
        print()
    finally:
        print("Goodbye!")


if __name__ == "__main__":
    main()
