"""LLM wrapper for OpenRouter (OpenAI-compatible API) with caching and retry logic.

All LLM calls are routed through OpenRouter (https://openrouter.ai) using the
OpenAI Python SDK pointed at OpenRouter's base URL. This lets us swap providers
and run different pipeline phases with different models simply by changing the
model strings in config.yaml under `models:` (e.g. "google/gemini-3.1-flash-lite",
"anthropic/claude-3.5-sonnet", "openai/gpt-5").
"""

import os
import json
import asyncio
import concurrent.futures
from typing import Dict, Any, Optional, List

from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

from .utils import get_cache_path, load_json, save_json, ensure_dir
from .cost_tracker import get_cost_tracker

# Load environment variables (OPENROUTER_API_KEY) from the repo-root .env if present.
load_dotenv()

# OpenRouter exposes an OpenAI-compatible REST API.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Default models. These are overridden per-phase from config.yaml (`models:`),
# but act as fallbacks when a model is not specified.
DEFAULT_MODEL = "google/gemini-3.1-flash-lite"
DEFAULT_EMBEDDING_MODEL = "google/gemini-embedding-2-preview"

# Optional attribution headers shown on OpenRouter dashboards (harmless to keep).
_DEFAULT_HEADERS = {
    "HTTP-Referer": "https://github.com/xandersteenbrugge/Choreo",
    "X-Title": "Choreo",
}

# Lazily-initialized singleton sync OpenRouter client (constructed after .env loads).
_client: Optional[OpenAI] = None


def _client_kwargs() -> Dict[str, Any]:
    """Shared constructor kwargs for the sync and async OpenRouter clients."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not found. Add it to the .env file at the repo root."
        )
    return {
        "base_url": OPENROUTER_BASE_URL,
        "api_key": api_key,
        "default_headers": _DEFAULT_HEADERS,
        # Bounded per-request time + no hidden SDK retries: one hung call must
        # fail fast into our own retry loop (max_retries/backoff above) instead
        # of pinning a batch at N-1/N for the SDK's 600s default × retries.
        # Same fix as the 2026-07-16 memory-engine sweep-timeout incident.
        "timeout": 120.0,
        "max_retries": 0,
    }


def get_openrouter_client() -> OpenAI:
    """Return a shared sync OpenRouter client (used for embeddings)."""
    global _client
    if _client is None:
        _client = OpenAI(**_client_kwargs())
    return _client


def make_async_openrouter_client() -> AsyncOpenAI:
    """Create a fresh async OpenRouter client; the caller must close it.

    A new client is created per async batch run rather than shared as a module
    singleton: the underlying httpx transport binds to the running event loop,
    and this pipeline opens a fresh event loop (``asyncio.run``) for each phase.
    """
    return AsyncOpenAI(**_client_kwargs())


def _get_nested(obj: Any, name: str) -> Any:
    """Read a field from an SDK pydantic object or a plain dict, tolerant of both.

    Non-standard OpenRouter fields (cost, cost_details, ...) land in the OpenAI
    SDK's ``model_extra`` rather than as declared attributes, so check there too.
    """
    if obj is None:
        return None
    val = getattr(obj, name, None)
    if val is None:
        extra = getattr(obj, "model_extra", None)
        if isinstance(extra, dict):
            val = extra.get(name)
    if val is None and isinstance(obj, dict):
        val = obj.get(name)
    return val


def extract_usage(response: Any) -> Dict[str, Any]:
    """Extract OpenRouter's usage accounting from a chat/embedding response.

    OpenRouter always returns full usage details — including the real ``cost``
    (in USD credits) — on every response, with no extra request param or API
    call needed. See:
    https://openrouter.ai/docs/cookbook/administration/usage-accounting

    Returns a dict with token counts, cost, and nested details (cached/reasoning
    tokens, upstream BYOK cost). ``cost`` is ``None`` only if not reported.
    """
    usage = getattr(response, "usage", None)

    prompt_tokens = int(_get_nested(usage, "prompt_tokens") or 0)
    completion_tokens = int(_get_nested(usage, "completion_tokens") or 0)
    total_tokens = int(_get_nested(usage, "total_tokens") or (prompt_tokens + completion_tokens))

    cost = _get_nested(usage, "cost")

    prompt_details = _get_nested(usage, "prompt_tokens_details")
    completion_details = _get_nested(usage, "completion_tokens_details")
    cost_details = _get_nested(usage, "cost_details")

    cached_tokens = int(_get_nested(prompt_details, "cached_tokens") or 0)
    reasoning_tokens = int(_get_nested(completion_details, "reasoning_tokens") or 0)
    upstream_cost = float(_get_nested(cost_details, "upstream_inference_cost") or 0.0)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost": float(cost) if cost is not None else None,
        "cached_tokens": cached_tokens,
        "reasoning_tokens": reasoning_tokens,
        "upstream_inference_cost": upstream_cost,
    }


def _build_extra_body(reasoning_effort: Optional[str]) -> Dict[str, Any]:
    """Assemble OpenRouter-specific request extensions.

    Note: OpenRouter usage accounting (cost + token details) is always on, so no
    ``usage: {include: true}`` flag is needed — it returns automatically.
    """
    extra_body: Dict[str, Any] = {}
    # Forward the reasoning effort whenever one is set (including "none" to turn
    # reasoning off). OpenRouter silently ignores this for non-reasoning models,
    # so it's safe to always send; reasoning models that can't be disabled would
    # otherwise default to "medium". A null/empty effort lets the model decide.
    if reasoning_effort:
        extra_body["reasoning"] = {"effort": reasoning_effort}
    return extra_body


def _build_chat_params(
    messages: List[Dict[str, str]],
    model: str,
    reasoning_effort: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Build the chat.completions request params shared by sync and async paths."""
    params: Dict[str, Any] = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
    }

    extra_body = _build_extra_body(reasoning_effort)
    if extra_body:
        params["extra_body"] = extra_body

    # Forward standard sampling params if explicitly provided.
    for key in ("temperature", "max_tokens", "top_p", "response_format"):
        if kwargs.get(key) is not None:
            params[key] = kwargs[key]

    return params


def _extract_message_content(response: Any) -> str:
    """Extract assistant text from an OpenAI-style chat completion response."""
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("No choices found in response")
    content = choices[0].message.content
    if content is None:
        raise ValueError("No message content found in response")
    return content


async def async_chat_completion(
    client: AsyncOpenAI,
    messages: List[Dict[str, str]],
    model: str,
    reasoning_effort: Optional[str] = None,
    **kwargs,
) -> Any:
    """Native async OpenRouter chat completion using a caller-provided client."""
    return await client.chat.completions.create(
        **_build_chat_params(messages, model, reasoning_effort, **kwargs)
    )


class JSONExtractionError(ValueError):
    """Raised when a model response cannot be parsed into JSON.

    Treated as retryable (see ``_is_retryable_error``): LLM output is stochastic,
    so re-sampling the same prompt usually yields valid JSON on the next attempt.
    """


def run_coro_blocking(coro):
    """Run an async coroutine to completion from synchronous code.

    From a plain sync context this is just ``asyncio.run`` (which also cancels
    any tasks still pending when its loop shuts down — no manual cleanup
    needed). When called from INSIDE a running event loop — e.g. choreo
    imported by an async web backend or MCP server — ``asyncio.run`` raises
    ``RuntimeError: cannot be called from a running event loop``, which the
    pipeline's broad exception handlers used to swallow, silently degrading to
    embedding-only scores. Here the coroutine instead runs on a fresh loop in a
    worker thread and we block on its result, so both contexts work.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class LLMWrapper:
    """Wrapper for LLM calls with caching and retries."""
    
    def __init__(self, cache_dir: Optional[str] = None, max_retries: int = 3, reasoning_effort: Optional[str] = "low",
                 max_concurrent_llm_calls: int = 16):
        # cache_dir=None disables the file-based response cache entirely
        # (in-memory pipeline runs, e.g. transient query matching).
        if cache_dir:
            self.cache_dir = ensure_dir(cache_dir) / "llm"
            ensure_dir(self.cache_dir)
        else:
            self.cache_dir = None
        self.max_retries = max_retries
        self.call_count = 0
        # Number of new responses written to the file cache this run. Lets an
        # adapter skip persistence work (e.g. Modal volume.commit) when a call
        # was served entirely from cache and nothing on disk changed.
        self.cache_writes = 0
        self.cost_tracker = get_cost_tracker()
        self.current_component = None  # Will be set by calling code
        # Default reasoning effort forwarded to OpenRouter (xhigh|high|medium|
        # low|minimal|none, or None to let the model decide). Used by any phase
        # that doesn't pass an explicit reasoning_effort; "low" is the
        # cost-effective baseline and is ignored on non-reasoning models.
        self.reasoning_effort = reasoning_effort
        # Global cap on how many LLM HTTP requests are in flight at once across
        # every batched phase. The dispatcher in batch_json_complete keeps
        # exactly this many calls running and fires the next the instant one
        # completes (semaphore-gated continuous dispatch), so this is the single
        # throughput/rate-limit knob. Set from config `concurrency:`.
        self.max_concurrent_llm_calls = max(1, int(max_concurrent_llm_calls))
        # Request provider-native JSON mode (response_format={"type":"json_object"})
        # on JSON completions so the model is forced to emit syntactically valid
        # JSON at the source. Auto-disabled for the rest of the run if a provider
        # rejects the param (see _async_json_complete_with_retry).
        self.json_mode = True
        # Async client, created/closed per batch_json_complete run (see note in
        # make_async_openrouter_client about per-event-loop lifecycle).
        self._async_client: Optional[AsyncOpenAI] = None

    def set_component(self, component: str):
        """Set the current component for cost tracking."""
        self.current_component = component

    def _record_usage(self, response: Any, model: str, call_type: str = "completion"):
        """Record token usage and cost for a completed call.

        Uses OpenRouter's native usage accounting (authoritative real cost in USD
        credits, returned automatically on every response).
        """
        try:
            usage = extract_usage(response)
            input_tokens = usage["prompt_tokens"]
            output_tokens = usage["completion_tokens"]

            cost = usage["cost"]
            if cost is None:
                # OpenRouter normally always reports cost; 0.0 is a safe default.
                print(f"⚠️  WARNING: OpenRouter reported no cost for {call_type} call with model {model}")
                cost = 0.0

            component = self.current_component if self.current_component else "unknown"
            self.cost_tracker.record_call(
                component=component,
                call_type=call_type,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                cached_tokens=usage["cached_tokens"],
                reasoning_tokens=usage["reasoning_tokens"],
                upstream_cost=usage["upstream_inference_cost"],
            )
        except (AttributeError, KeyError, TypeError) as e:
            print(f"⚠️  WARNING: Could not track cost for {call_type} call with model {model}: {e}")
    
    def _prepare_json_prompt(self, prompt: str, schema_hint: Optional[str]) -> str:
        """Prepare prompt for JSON output."""
        json_instruction = "Respond with valid JSON only. No additional text or explanations."
        
        if schema_hint:
            json_instruction += f"\nExpected JSON structure: {schema_hint}"
        
        return f"{prompt}\n\n{json_instruction}"
    
    @staticmethod
    def _first_balanced_json(text: str) -> Optional[Any]:
        """Parse the first balanced ``{...}`` / ``[...]`` span in ``text``.

        Scans for the first opening brace/bracket and walks to its matching close,
        correctly skipping braces that appear inside string literals (and their
        escapes). This recovers a JSON object even when the model wraps it in
        prose ("Here is the JSON: {...} Hope that helps!") or pretty-prints it
        across many lines — cases the old line-by-line scan could not handle.
        Returns the parsed value, or None if no balanced span parses.
        """
        for start, opener in enumerate(text):
            if opener not in "{[":
                continue
            closer = "}" if opener == "{" else "]"
            depth = 0
            in_str = False
            escaped = False
            for end in range(start, len(text)):
                ch = text[end]
                if in_str:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            # strict=False tolerates literal control chars
                            # (newlines/tabs) inside string values — common in
                            # multi-paragraph prose responses.
                            return json.loads(text[start:end + 1], strict=False)
                        except json.JSONDecodeError:
                            break  # try the next opener
        return None

    def _extract_json(self, response: str) -> Dict[str, Any]:
        """Extract JSON from a model response, tolerant of common wrappers.

        Order: direct parse → fenced code block (```json / ```) → first balanced
        JSON span anywhere in the text. Raises JSONExtractionError (retryable) if
        nothing parses, so the caller re-samples rather than silently dropping
        the profile to empty sections.
        """
        response = response.strip()

        # Try direct JSON parse first. strict=False allows literal control
        # characters (unescaped newlines/tabs) inside string values — models
        # routinely emit these in multi-paragraph prose, and strict parsing
        # would otherwise reject structurally-valid JSON and force a re-sample.
        try:
            return json.loads(response, strict=False)
        except json.JSONDecodeError:
            pass

        # Look for JSON within markdown code blocks (```json or plain ```)
        for fence in ("```json", "```"):
            if fence in response:
                start = response.find(fence) + len(fence)
                end = response.find("```", start)
                if end > start:
                    try:
                        return json.loads(response[start:end].strip(), strict=False)
                    except json.JSONDecodeError:
                        pass

        # Fall back to the first balanced JSON span anywhere in the text.
        obj = self._first_balanced_json(response)
        if obj is not None:
            return obj

        raise JSONExtractionError(
            f"Could not extract valid JSON from response: {response[:200]}..."
        )
    
    def get_stats(self) -> Dict[str, int]:
        """Get usage statistics."""
        return {"total_calls": self.call_count}

    async def batch_json_complete(
        self,
        prompts: List[str],
        model: str,
        cache_keys: Optional[List[Optional[str]]] = None,
        schema_hints: Optional[List[Optional[str]]] = None,
        max_concurrent: Optional[int] = None,
        max_retries: int = 3,
        reasoning_effort: Optional[str] = None,
        retry_delay_base: float = 1.0,
        verbosity: int = 0,
        print_reasoning_summary: bool = False,
        max_tokens: Optional[int] = None,
        progress_label: Optional[str] = None,
        progress_interval: float = 10.0,
        deadline_s: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple prompts concurrently with rate-limit retry handling.

        Dispatch is semaphore-gated: at most ``max_concurrent`` calls run at any
        instant, and the next prompt fires the moment any in-flight call returns
        — no fire-a-batch-then-wait barrier — so the slowest call in a notional
        batch never stalls the others.

        Args:
            prompts: List of prompts to process
            model: Model name (e.g., "gpt-4o-mini")
            cache_keys: Optional list of cache keys (if None, no caching)
            schema_hints: Optional list of schema hints
            max_concurrent: Max concurrent in-flight requests. None inherits the
                wrapper's ``max_concurrent_llm_calls`` (config `concurrency:`).
            max_retries: Maximum retries for rate limit errors (default: 3)
            retry_delay_base: Base delay for exponential backoff (default: 1.0s)
            deadline_s: Optional wall-clock budget for the whole batch. None
                (default) waits for every call. When set, calls still in flight
                at the deadline are cancelled and their slots stay ``None``, so
                the caller sees "no answer" for those prompts and decides what
                that means. Use it when the model's TAIL latency costs more than
                the last few answers are worth — but note cancelled calls still
                burn the tokens the provider already generated.

        Returns:
            List of parsed JSON responses in same order as input prompts. A slot
            is ``None`` if its call was cancelled at ``deadline_s``, and holds an
            Exception if the call failed.
        """
        if not prompts:
            return []

        # Fall back to the wrapper's configured default when no per-phase
        # effort is given (None means "inherit the config default").
        if reasoning_effort is None:
            reasoning_effort = self.reasoning_effort

        # None means "inherit the wrapper default" (set from config concurrency:).
        if max_concurrent is None:
            max_concurrent = self.max_concurrent_llm_calls
        max_concurrent = max(1, int(max_concurrent))

        n_prompts = len(prompts)
        cache_keys = cache_keys or [None] * n_prompts
        schema_hints = schema_hints or [None] * n_prompts

        print(f"Processing {n_prompts} prompts with up to {max_concurrent} concurrent LLM calls")
        
        # Check cache first and prepare uncached tasks
        results = [None] * n_prompts
        uncached_indices = []
        
        for i, (prompt, cache_key, schema_hint) in enumerate(zip(prompts, cache_keys, schema_hints)):
            if cache_key and self.cache_dir:
                cache_path = get_cache_path(self.cache_dir, cache_key)
                if cache_path.exists():
                    try:
                        results[i] = load_json(cache_path)
                        continue
                    except Exception as e:
                        print(f"Warning: Failed to load cache {cache_path}: {e}")
            uncached_indices.append(i)
        
        cached_count = n_prompts - len(uncached_indices)
        if cached_count > 0:
            print(f"Found {cached_count} cached results")

        if not uncached_indices:
            return results

        # Open one async client for this run (bound to the current event loop)
        # and close it when done — see make_async_openrouter_client.
        self._async_client = make_async_openrouter_client()
        try:
            # Semaphore-gated continuous dispatch: every uncached prompt becomes
            # a task immediately, but the semaphore lets only `max_concurrent`
            # actually hit the API at once. As soon as any call returns it
            # releases its slot and the next queued task acquires it — so the
            # provider is kept saturated at exactly `max_concurrent` without the
            # slowest call in a window stalling the rest (the old fire-a-batch-
            # then-await behavior).
            semaphore = asyncio.Semaphore(max_concurrent)
            total = len(uncached_indices)
            completed = 0

            async def _run_one(i: int):
                nonlocal completed
                async with semaphore:
                    try:
                        result = await self._async_json_complete_with_retry(
                            prompt=prompts[i],
                            model=model,
                            cache_key=cache_keys[i],
                            schema_hint=schema_hints[i],
                            max_retries=max_retries,
                            retry_delay_base=retry_delay_base,
                            reasoning_effort=reasoning_effort,
                            verbosity=verbosity,
                            print_reasoning_summary=print_reasoning_summary,
                            max_tokens=max_tokens,
                        )
                    except Exception as e:  # keep going; caller inspects per-item
                        result = e
                completed += 1
                if isinstance(result, Exception):
                    print(f"Task failed for prompt {i} ({completed}/{total}): {result}")
                elif verbosity > 0:
                    print(f"Completed {completed}/{total} LLM calls")
                # Return the index alongside the result so out-of-order
                # completion still maps back to the right slot.
                return i, result

            # Periodic progress ticker: prints the completed fraction every
            # `progress_interval` seconds while the batch runs, so long phases
            # show liveness instead of going silent until the final line. It
            # only reads `completed`/`total`, never touches the result slots,
            # and is cancelled the moment all tasks finish.
            label = f"[{progress_label}] " if progress_label else ""

            async def _progress_ticker():
                try:
                    while True:
                        await asyncio.sleep(progress_interval)
                        if completed >= total:
                            break
                        pct = 100.0 * completed / total if total else 100.0
                        print(f"{label}LLM progress: {completed}/{total} ({pct:.0f}%)")
                except asyncio.CancelledError:
                    pass

            tasks = [asyncio.create_task(_run_one(i)) for i in uncached_indices]
            ticker = asyncio.create_task(_progress_ticker())
            try:
                # timeout=None waits for every task (plain-gather semantics).
                # With a deadline, whatever has landed by then is kept and the
                # stragglers are cancelled — their slots stay None.
                done, pending = await asyncio.wait(tasks, timeout=deadline_s)
                for task in done:
                    i, result = task.result()
                    results[i] = result  # exceptions kept in-place for caller to handle
                if pending:
                    for task in pending:
                        task.cancel()
                    # Let the cancellations settle before the client closes.
                    await asyncio.gather(*pending, return_exceptions=True)
                    print(
                        f"{label}Deadline {deadline_s:g}s reached at "
                        f"{completed}/{total} complete — cancelled {len(pending)} "
                        f"in-flight call(s); their slots stay unanswered"
                    )
            finally:
                ticker.cancel()
        finally:
            await self._async_client.close()
            self._async_client = None

        print(f"Completed processing all {n_prompts} prompts")
        await cleanup_background_tasks()
        return results
    
    async def _async_json_complete_with_retry(
        self,
        prompt: str,
        model: str,
        cache_key: Optional[str] = None,
        schema_hint: Optional[str] = None,
        max_retries: int = 3,
        retry_delay_base: float = 1.0,
        reasoning_effort: Optional[str] = None,
        verbosity: int = 0,
        print_reasoning_summary: bool = False,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Single async JSON completion with retry logic."""
        # Check cache first
        if cache_key and self.cache_dir:
            cache_path = get_cache_path(self.cache_dir, cache_key)
            if cache_path.exists():
                try:
                    return load_json(cache_path)
                except Exception as e:
                    print(f"Warning: Failed to load cache {cache_path}: {e}")
        
        json_prompt = self._prepare_json_prompt(prompt, schema_hint)
        
        for attempt in range(max_retries + 1):
            try:
                if verbosity > 1:
                    print("-----------------------------------------------------------")
                    print(f"Calling LLM {model} with prompt:\n{json_prompt}")
                    print("-----------------------------------------------------------")

                call_kwargs: Dict[str, Any] = {}
                if self.json_mode:
                    call_kwargs["response_format"] = {"type": "json_object"}
                if max_tokens is not None:
                    call_kwargs["max_tokens"] = max_tokens

                response = await async_chat_completion(
                    self._async_client,
                    messages=[{"role": "user", "content": json_prompt}],
                    model=model,
                    reasoning_effort=reasoning_effort if reasoning_effort is not None else self.reasoning_effort,
                    **call_kwargs,
                )

                # Track token usage and cost.
                self._record_usage(response, model, "completion")

                if verbosity > 1:
                    print(f"Response: {response}")

                # Print reasoning trace if requested and the model returned one.
                if print_reasoning_summary:
                    reasoning = getattr(response.choices[0].message, "reasoning", None)
                    if reasoning:
                        print(f"🧠 Reasoning: {str(reasoning)[:500]}")

                # Extract assistant text from the chat completion response.
                content = _extract_message_content(response)

                # Parse JSON
                result = self._extract_json(content.strip())
                
                # Cache result
                if cache_key and self.cache_dir:
                    try:
                        cache_path = get_cache_path(self.cache_dir, cache_key)
                        save_json(result, cache_path)
                        self.cache_writes += 1
                    except Exception as e:
                        print(f"Warning: Failed to save cache {cache_path}: {e}")
                
                # Update call count (thread-safe increment)
                self.call_count += 1
                
                return result
                
            except Exception as e:
                # If the provider rejects JSON mode, disable it for the rest of
                # the run and retry without it (the hardened parser still copes).
                if self.json_mode and "response_format" in str(e).lower():
                    print(f"⚠️  Provider rejected JSON mode (response_format); disabling for this run: {e}")
                    self.json_mode = False
                    retryable = True
                else:
                    retryable = self._is_retryable_error(e)

                if attempt < max_retries and retryable:
                    delay = retry_delay_base * (2 ** attempt)
                    print(f"Transient LLM error, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    print(f"Failed after {attempt + 1} attempts: {e}")
                    raise
        
        # Should not reach here
        raise RuntimeError("Unexpected end of retry loop")
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is related to rate limiting."""
        error_str = str(error).lower()
        rate_limit_indicators = [
            "rate limit", "rate_limit", "429", "too many requests",
            "quota exceeded", "rate exceeded", "throttle", "throttled"
        ]
        return any(indicator in error_str for indicator in rate_limit_indicators)

    def _is_retryable_error(self, error: Exception) -> bool:
        """Whether an error is transient and worth retrying.

        Covers rate limits plus transient server (5xx) and connection/timeout
        errors — common on live venue wifi — so a single blip doesn't drop a
        profile to empty sections or a pair to embed-only scoring. A 4xx client
        error other than 429 (e.g. a malformed request) is NOT retried; it fails
        fast so it stays visible.
        """
        # Malformed-JSON responses are stochastic — re-sampling usually fixes it.
        if isinstance(error, JSONExtractionError):
            return True

        if self._is_rate_limit_error(error):
            return True

        # Honor an explicit HTTP status code if the SDK exposes one.
        status = getattr(error, "status_code", None)
        if isinstance(status, int):
            if status == 429 or 500 <= status < 600:
                return True
            if 400 <= status < 500:
                return False

        error_str = str(error).lower()
        transient_indicators = [
            "500", "502", "503", "504", "520", "521", "522", "524", "529",
            "internal server error", "bad gateway", "service unavailable",
            "gateway timeout", "overloaded", "timeout", "timed out",
            "connection", "connection error", "connection reset", "temporarily",
        ]
        return any(indicator in error_str for indicator in transient_indicators)


async def cleanup_background_tasks():
    """Cancel any lingering background tasks to ensure clean exit."""
    pending = asyncio.all_tasks()
    current = asyncio.current_task()
    background_tasks = [task for task in pending if task != current and not task.done()]
    
    if background_tasks:
        print(f"Cleaning up {len(background_tasks)} background tasks")
        for task in background_tasks:
            task.cancel()
        await asyncio.sleep(0.1)  # Give tasks time to cancel