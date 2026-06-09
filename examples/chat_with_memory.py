"""Chat with an AI assistant that remembers your conversations.

Prerequisites:
    pip install infini-memory openai

    export OPENAI_API_KEY="sk-..."

Usage:
    python examples/chat_with_memory.py
"""

import atexit
import readline  # noqa: F401  # enable line editing (arrow keys, delete) in input()
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from infini_memory import Memory

openai_client = OpenAI()
memory = Memory(search_strategy="BM25_partition")

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
# The per-turn memory system message is NOT stored here — it's injected
# transiently for the LLM call and never persisted to history or to memory.add().
conversation_history: list[dict] = []


def _add_memory_async(messages: list, user_id: str, seq: int) -> None:
    def _run() -> None:
        try:
            memory.add(messages, user_id=user_id, seq=seq)
        except Exception as e:
            print(f"\n[memory.add failed in background] {e}")

    _memory_executor.submit(_run)


def _shutdown_executor() -> None:
    # Wait for pending memory writes to finish before the process exits so
    # data from the final turn isn't lost.
    _memory_executor.shutdown(wait=True)


atexit.register(_shutdown_executor)


def _format_memories(relevant_memories) -> str:
    """Build the memory block for the system prompt.

    Each result may carry a `summary` (set for split docs) and/or a `content`
    field. The CURRENT document — where the latest turns are appended — has
    an empty summary, so falling back to `content` is what makes a fresh
    chat session actually recall previous turns.
    """
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


def _build_memory_system_message(message: str, user_id: str) -> dict | None:
    """Search memories and wrap them as a transient system message.

    Returns None when no memories are found, so the LLM call doesn't carry
    an empty memory block.
    """
    relevant_memories = memory.search(query=message, user_id=user_id)
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


def chat_with_memories(message: str, user_id: str = "default_user") -> str:
    global _seq_counter
    _seq_counter += 1
    seq = _seq_counter
    _save_seq(seq)

    user_msg = {"role": "user", "content": message}

    # Build the request: base system + transient memory system + history + new user.
    # The memory system message is created fresh each turn and discarded after.
    request_messages: list[dict] = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]
    memory_sys_msg = _build_memory_system_message(message, user_id)
    if memory_sys_msg is not None:
        request_messages.append(memory_sys_msg)
    request_messages.extend(conversation_history)
    request_messages.append(user_msg)

    stream = openai_client.chat.completions.create(
        model="gpt-5-mini", messages=request_messages, stream=True
    )
    chunks = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            chunks.append(delta)
            sys.stdout.write(delta)
            sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    assistant_response = "".join(chunks)
    assistant_msg = {"role": "assistant", "content": assistant_response}

    # Append only the real turn (no memory system message) to history,
    # and persist only the real turn to memory so the next extraction
    # doesn't re-ingest previously retrieved memories.
    conversation_history.append(user_msg)
    conversation_history.append(assistant_msg)
    _add_memory_async([user_msg, assistant_msg], user_id, seq)

    return assistant_response


def main():
    print("Chat with AI (type 'exit' to quit)")
    print()
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        sys.stdout.write("AI: ")
        sys.stdout.flush()
        chat_with_memories(user_input)
        print()


if __name__ == "__main__":
    main()
