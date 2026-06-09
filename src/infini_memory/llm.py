from __future__ import annotations

from typing import Any, List, Dict, Optional
import json
import logging
import time
import random


class LLMClient:
    """Minimal LLM call adapter layer.

    Provides an OpenAI SDK call interface without placeholder implementations.
    """

    # Models that do not support the temperature parameter (only support default value 1)
    MODELS_WITHOUT_TEMPERATURE = ["gpt-5-mini"]

    def __init__(
        self,
        caller: Any | None = None,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        retry_max_attempts: int = 10,
        retry_initial_wait: float = 5.0,
        retry_max_wait: float = 60.0,
        retry_jitter: float = 0.2,
    ):
        self._caller = caller
        self.api_key = (api_key or "").strip()
        self.base_url = (base_url or "").strip()
        self.logger = logging.getLogger("infini_memory")
        self.retry_max_attempts = retry_max_attempts
        self.retry_initial_wait = retry_initial_wait
        self.retry_max_wait = retry_max_wait
        self.retry_jitter = retry_jitter

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        # Log request (debug level)
        try:
            msg_json = json.dumps(messages, ensure_ascii=False, indent=2)
        except Exception:
            msg_json = str(messages)
        endpoint = (self.base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        self.logger.debug(
            "LLM request: model=%s temperature=%s base_url=%s endpoint=%s\n====== LLM INPUT ======\n%s\n====== END INPUT ======",
            model,
            temperature,
            self.base_url or "<default>",
            endpoint,
            msg_json,
        )

        for attempt in range(1, self.retry_max_attempts + 1):
            try:
                out = self._chat_once(messages=messages, model=model, temperature=temperature)
                self.logger.debug("====== LLM OUTPUT ======\n%s\n====== END OUTPUT ======", out)
                return out
            except Exception as e:
                if attempt < self.retry_max_attempts:
                    wait_time = min(self.retry_initial_wait * (2 ** (attempt - 1)), self.retry_max_wait)
                    if self.retry_jitter:
                        jitter_ratio = random.uniform(-self.retry_jitter, self.retry_jitter)
                        wait_time = max(0.0, wait_time * (1 + jitter_ratio))
                    self.logger.warning(
                        "LLM call failed, retry %s/%s will occur in %.2f seconds: %s",
                        attempt,
                        self.retry_max_attempts,
                        wait_time,
                        e,
                    )
                    time.sleep(wait_time)
                    continue
                raise RuntimeError(f"LLM call failed: {e}") from e

        raise RuntimeError(f"LLM call failed (retried {self.retry_max_attempts} times)")

    def _chat_once(
        self,
        *,
        messages: List[Dict[str, str]],
        model: str | None,
        temperature: float | None,
    ) -> str:
        # 1) Custom callback takes priority
        if callable(self._caller):
            self.logger.debug("LLM branch: using custom caller")
            return str(self._caller(messages=messages, model=model, temperature=temperature))

        # 2) OpenAI new SDK (openai>=1.0)
        if self.api_key and model:
            try:
                from openai import OpenAI as OpenAIClient  # type: ignore

                client = OpenAIClient(api_key=self.api_key, base_url=self.base_url or None)
                # Build API call parameters
                api_kwargs = {
                    "model": model,
                    "messages": messages,
                }
                # Check if the model supports the temperature parameter
                if temperature is not None and model not in self.MODELS_WITHOUT_TEMPERATURE:
                    api_kwargs["temperature"] = temperature

                api_start_time = time.time()
                resp = client.chat.completions.create(**api_kwargs)
                api_elapsed = time.time() - api_start_time
                content = resp.choices[0].message.content if resp and resp.choices else ""
                self.logger.debug("LLM branch: OpenAI new SDK")
                self.logger.info("LLM API call duration: %.1f seconds (model=%s)", api_elapsed, model)
                # Detect empty response -- when API returns error code (e.g. quota exceeded),
                # choices is None causing content to be empty, need to raise exception for retry
                if not content:
                    error_code = getattr(resp, "code", None)
                    error_msg = getattr(resp, "msg", None) or getattr(resp, "message", None) or ""
                    self.logger.warning("LLM API returned empty response! resp=%s", resp)
                    if error_code or error_msg:
                        raise RuntimeError(f"LLM API returned empty response (code={error_code}, msg={error_msg})")
                    # Even without error code, empty response should be retried
                    raise RuntimeError("LLM API returned empty response")
                # Request ID and usage (if available)
                try:
                    req_id = getattr(resp, "id", None)
                    usage = getattr(resp, "usage", None)
                    if req_id:
                        self.logger.debug("LLM request ID: %s", req_id)
                    if usage:
                        # usage may contain prompt_tokens/completion_tokens/total_tokens
                        self.logger.debug("LLM usage: %s", usage)
                except Exception:
                    ...
                return content.strip()
            except ImportError:
                raise RuntimeError("openai package required: pip install openai")
            except Exception:
                raise

        # 3) OpenAI legacy SDK (openai<1.0)
        if self.api_key and model:
            try:
                import openai  # type: ignore

                openai.api_key = self.api_key
                if self.base_url:
                    # Legacy SDK base_url field may be openai.base_url or openai.api_base
                    setattr(openai, "base_url", self.base_url)
                    setattr(openai, "api_base", self.base_url)
                # Build API call parameters
                api_kwargs = {
                    "model": model,
                    "messages": messages,
                }
                # Check if the model supports the temperature parameter
                if temperature is not None and model not in self.MODELS_WITHOUT_TEMPERATURE:
                    api_kwargs["temperature"] = temperature

                api_start_time = time.time()
                resp = openai.ChatCompletion.create(**api_kwargs)
                api_elapsed = time.time() - api_start_time
                content = resp["choices"][0]["message"]["content"] if resp and resp.get("choices") else ""
                self.logger.debug("LLM branch: OpenAI legacy SDK")
                self.logger.info("LLM API call duration: %.1f seconds (model=%s)", api_elapsed, model)
                # Detect empty response -- when API returns error code, choices may be empty, need to raise exception for retry
                if not content:
                    error_code = resp.get("code") if hasattr(resp, "get") else None
                    error_msg = resp.get("msg") or resp.get("message", "") if hasattr(resp, "get") else ""
                    self.logger.warning("LLM API returned empty response! resp=%s", resp)
                    if error_code or error_msg:
                        raise RuntimeError(f"LLM API returned empty response (code={error_code}, msg={error_msg})")
                    raise RuntimeError("LLM API returned empty response")
                try:
                    req_id = resp.get("id") if hasattr(resp, "get") else None
                    usage = resp.get("usage") if hasattr(resp, "get") else None
                    if req_id:
                        self.logger.debug("LLM request ID: %s", req_id)
                    if usage:
                        self.logger.debug("LLM usage: %s", usage)
                except Exception:
                    ...
                return content.strip()
            except ImportError:
                raise RuntimeError("openai package required: pip install openai")
            except Exception:
                raise

        # 4) No available way to call LLM
        if not self.api_key:
            raise RuntimeError("LLM call failed: api_key not configured")
        if not model:
            raise RuntimeError("LLM call failed: model not specified")
        raise RuntimeError("LLM call failed: no LLM implementation available")


def _clip(text: str, limit: int = 2000) -> str:
    """Limit log output length to avoid polluting log files."""
    try:
        t = str(text)
        return t if len(t) <= limit else (t[:limit] + "...<clipped>")
    except Exception:
        return "<unprintable>"


__all__ = ["LLMClient"]
