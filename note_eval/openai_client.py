import json
import os
import time
import threading
from collections import deque
from typing import Any, Callable, Dict, Iterable, List, Optional
import numpy as np
from numpy.typing import NDArray
from openai import OpenAI


class OpenAIClient:
    """
    Wrapper around OpenAI API with token usage tracking and endpoint-specific rate limiting.
    """

    # Separate, process-wide RPM buckets for responses and embeddings
    _resp_lock = threading.Lock()
    _resp_times = deque()   # timestamps for responses.create (RPM)
    _resp_tok_lock = threading.Lock()
    _resp_tok_times = deque()  # (timestamp, tokens) for responses TPM
    _emb_lock = threading.Lock()
    _emb_times = deque()    # timestamps for embeddings.create (RPM)
    _emb_tok_lock = threading.Lock()
    _emb_tok_times = deque()   # (timestamp, tokens) for embeddings TPM

    def __init__(self, api_key: str, embedding_model: str = "text-embedding-3-small") -> None:
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key
            embedding_model: OpenAI embedding model to use
        """
        
        self.client = OpenAI(api_key=api_key)
        self.embedding_model = embedding_model
        
        # Token usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self.failed_calls = 0

        # Endpoint-specific RPM limits (can be overridden per env)
        # If only OPENAI_RPM_LIMIT is set, it applies to both as a fallback.
        def read_limit(env_key: str, fallback_env: str, default_val: int) -> int:
            try:
                v = int(os.getenv(env_key, os.getenv(fallback_env, str(default_val))))
                return v if v > 0 else default_val
            except Exception:
                return default_val

        self._rpm_limit_responses = read_limit("OPENAI_RPM_LIMIT_RESPONSES", "OPENAI_RPM_LIMIT", 500)
        self._rpm_limit_embeddings = read_limit("OPENAI_RPM_LIMIT_EMBEDDINGS", "OPENAI_RPM_LIMIT", 500)

        # Tokens-per-minute limits and output estimates
        self._tpm_limit_responses = read_limit("OPENAI_TPM_LIMIT_RESPONSES", "OPENAI_TPM_LIMIT", 200000)
        self._tpm_limit_embeddings = read_limit("OPENAI_TPM_LIMIT_EMBEDDINGS", "OPENAI_TPM_LIMIT", 3000000)
        try:
            self._resp_output_estimate = int(os.getenv("OPENAI_TPM_ESTIMATE_OUTPUT_RESPONSES", "256"))
            if self._resp_output_estimate < 0:
                self._resp_output_estimate = 0
        except Exception:
            self._resp_output_estimate = 256

    @classmethod
    def _acquire_slot(cls, kind: str, rpm_limit: int) -> None:
        """Block until a request slot is available for the given endpoint kind."""
        lock = cls._resp_lock if kind == "responses" else cls._emb_lock
        times = cls._resp_times if kind == "responses" else cls._emb_times
        while True:
            now = time.time()
            with lock:
                window_start = now - 60.0
                while times and times[0] < window_start:
                    times.popleft()
                if len(times) < rpm_limit:
                    times.append(now)
                    return
                earliest = times[0]
                sleep_s = max(0.0, (earliest + 60.0) - now)
            time.sleep(min(sleep_s, 1.0))

    @classmethod
    def _estimate_tokens(cls, text: str) -> int:
        # Rough heuristic: ~4 chars per token
        if not text:
            return 0
        return max(1, int(len(text) / 4))

    @classmethod
    def _acquire_tpm(cls, kind: str, tpm_limit: int, est_tokens: int) -> None:
        """Block until adding est_tokens keeps tokens within the last 60s <= tpm_limit."""
        tok_lock = cls._resp_tok_lock if kind == "responses" else cls._emb_tok_lock
        tok_times = cls._resp_tok_times if kind == "responses" else cls._emb_tok_times
        while True:
            now = time.time()
            with tok_lock:
                window_start = now - 60.0
                # Drop old entries
                while tok_times and tok_times[0][0] < window_start:
                    tok_times.popleft()
                used = 0
                for _, tks in tok_times:
                    used += tks
                if used + est_tokens <= tpm_limit:
                    tok_times.append((now, est_tokens))
                    return
                # Sleep until oldest entry expires sufficiently
                earliest_ts = tok_times[0][0]
                sleep_s = max(0.0, (earliest_ts + 60.0) - now)
            time.sleep(min(sleep_s, 1.0))

    def encode(
        self,
        texts: Iterable[str],
        model: Optional[str] = None,
        normalize: bool = True,
    ) -> NDArray[np.float32]:
        """
        Create embeddings for a list of texts using OpenAI embeddings API.

        Args:
            texts: Iterable of strings to embed
            model: Optional model override; defaults to self.embedding_model
            normalize: Whether to L2-normalize rows (cosine-ready)

        Returns:
            np.ndarray of shape (N, D) float32
        """
        inputs = list(texts)
        if not inputs:
            # Return empty matrix with default dimensionality
            return np.zeros((0, 1536), dtype=np.float32)

        def do_embed():
            # Rate limit across instances/threads (embeddings bucket)
            self._acquire_slot("embeddings", self._rpm_limit_embeddings)
            # TPM for embeddings: estimate by input chars
            est_tokens_local = 0
            try:
                for t in inputs:
                    est_tokens_local += self._estimate_tokens(t)
            except Exception:
                est_tokens_local = self._estimate_tokens("".join(inputs)) if isinstance(inputs, list) else self._estimate_tokens(str(inputs))
            self._acquire_tpm("embeddings", self._tpm_limit_embeddings, est_tokens_local)
            return self.client.embeddings.create(
                input=inputs,
                model=model or self.embedding_model,
            )

        # Retry embeddings with backoff to handle transient/rate-limit errors
        resp = self.with_backoff(do_embed, max_tries=3, base=1.5)
        arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
        if not normalize:
            return arr
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
        return (arr / norms).astype(np.float32)
        
    def chat_completion_json(
        self,
        messages: List[Dict[str, str]],
        model: str,
    ) -> Dict[str, Any]:
        """
        Call OpenAI API and extract JSON from response.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name to use
            
        Returns:
            Parsed JSON object from the response
            
        Raises:
            Exception if the API call fails after retry
        """
        self.total_calls += 1
        
        # Format messages into a single input string for the responses API
        input_text = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                input_text += f"System: {content}\n\n"
            elif role == "user":
                input_text += f"User: {content}\n\n"
            elif role == "assistant":
                input_text += f"Assistant: {content}\n\n"
        
        try:
            # Rate limit across instances/threads (responses bucket)
            self._acquire_slot("responses", self._rpm_limit_responses)
            # TPM for responses: estimate input + expected output
            est_in = self._estimate_tokens(input_text)
            est_total = est_in + self._resp_output_estimate
            self._acquire_tpm("responses", self._tpm_limit_responses, est_total)
            response = self.client.responses.create(
                model=model,
                input=input_text.strip(),
                temperature=0.0,
            )
            
            # Track token usage if available
            if hasattr(response, 'usage'):
                in_tok = getattr(response.usage, 'input_tokens', 0)
                out_tok = getattr(response.usage, 'output_tokens', 0)
                self.total_input_tokens += in_tok
                self.total_output_tokens += out_tok
                # Correct TPM estimation drift by recording delta
                try:
                    actual_total = int(in_tok) + int(out_tok)
                    correction = actual_total - est_total
                    if correction != 0:
                        now = time.time()
                        with self._resp_tok_lock:
                            self._resp_tok_times.append((now, correction))
                except Exception:
                    pass
            
            # Extract text from response
            output_text = ""
            for item in response.output:
                if hasattr(item, "content"):
                    for content in item.content:
                        if hasattr(content, "text"):
                            output_text += content.text
            
            return self._coerce_json_from_text(output_text)
            
        except Exception as e:
            # Retry with explicit JSON instruction
            try:
                retry_input = input_text + "\n\nReturn ONLY valid JSON with the specified keys. No prose."
                # Re-acquire a slot before retry (responses bucket)
                self._acquire_slot("responses", self._rpm_limit_responses)
                # Also gate by TPM on retry to avoid under-throttling
                # Reserve the same estimated tokens as the first attempt; conservative but safe
                self._acquire_tpm("responses", self._tpm_limit_responses, est_total)
                response = self.client.responses.create(
                    model=model,
                    input=retry_input,
                    temperature=0.0,
                )
                
                # Track token usage if available
                if hasattr(response, 'usage'):
                    in_tok = getattr(response.usage, 'input_tokens', 0)
                    out_tok = getattr(response.usage, 'output_tokens', 0)
                    self.total_input_tokens += in_tok
                    self.total_output_tokens += out_tok
                    # Correct TPM estimation drift
                    try:
                        actual_total = int(in_tok) + int(out_tok)
                        correction = actual_total - (est_in + self._resp_output_estimate)
                        if correction != 0:
                            now = time.time()
                            with self._resp_tok_lock:
                                self._resp_tok_times.append((now, correction))
                    except Exception:
                        pass
                
                output_text = ""
                for item in response.output:
                    if hasattr(item, "content") and item.content is not None:
                        for content in item.content:
                            if hasattr(content, "text"):
                                output_text += content.text
                
                return self._coerce_json_from_text(output_text)
            except Exception as retry_error:
                self.failed_calls += 1
                raise retry_error
    
    def with_backoff(self, fn: Callable[[], Any], max_tries: int = 3, base: float = 1.5) -> Any:
        """
        Execute a function with exponential backoff on failure.
        
        Args:
            fn: Function to execute
            max_tries: Maximum number of attempts
            base: Base multiplier for exponential backoff
            
        Returns:
            Result of the function call
            
        Raises:
            Exception from the last failed attempt
        """
        for i in range(max_tries):
            try:
                return fn()
            except Exception as e:
                if i == max_tries - 1:
                    self.failed_calls += 1
                    raise
                time.sleep(base * (2 ** i))
    
    def get_token_usage(self) -> Dict[str, int]:
        """
        Get current token usage statistics.
        
        Returns:
            Dictionary with token usage information
        """
        return {
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "successful_calls": self.total_calls - self.failed_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total cost w/ GPT-4.1-mini (in $)": self.total_input_tokens * 0.4 / 1000000 + self.total_output_tokens * 1.6 / 1000000,
        }
    
    def reset_token_usage(self) -> None:
        """Reset all token usage counters to zero."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self.failed_calls = 0
    
    @staticmethod
    def _coerce_json_from_text(content_text: str) -> Dict[str, Any]:
        """
        Extract and parse JSON from text response.
        
        Handles various formats including code blocks.
        
        Args:
            content_text: Text content from API response
            
        Returns:
            Parsed JSON object
            
        Raises:
            ValueError if content is empty
            json.JSONDecodeError if JSON is invalid
        """
        s = (content_text or "").strip()
        if not s:
            raise ValueError("empty content")
        
        # Strip triple-fenced code blocks like ```json ... ``` or ``` ... ```
        if s.startswith("```"):
            parts = s.split("\n", 1)
            s = parts[1] if len(parts) > 1 else ""
            if s.endswith("```"):
                s = s.rsplit("```", 1)[0]
            s = s.strip()
        
        return json.loads(s)
