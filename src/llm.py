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
from typing import Dict, Any, Optional, List

from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

from utils import get_cache_path, load_json, save_json, ensure_dir
from cost_tracker import get_cost_tracker

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


def _build_extra_body(reasoning_effort: Optional[str], enable_reasoning: bool) -> Dict[str, Any]:
    """Assemble OpenRouter-specific request extensions.

    Note: OpenRouter usage accounting (cost + token details) is always on, so no
    ``usage: {include: true}`` flag is needed — it returns automatically.
    """
    extra_body: Dict[str, Any] = {}
    # Only forward a reasoning effort to reasoning-capable models when enabled;
    # the default (gemini-flash-lite) is not a reasoning model.
    if enable_reasoning and reasoning_effort:
        extra_body["reasoning"] = {"effort": reasoning_effort}
    return extra_body


def _build_chat_params(
    messages: List[Dict[str, str]],
    model: str,
    reasoning_effort: Optional[str] = None,
    enable_reasoning: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Build the chat.completions request params shared by sync and async paths."""
    params: Dict[str, Any] = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
    }

    extra_body = _build_extra_body(reasoning_effort, enable_reasoning)
    if extra_body:
        params["extra_body"] = extra_body

    # Forward standard sampling params if explicitly provided.
    for key in ("temperature", "max_tokens", "top_p"):
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
    enable_reasoning: bool = False,
    **kwargs,
) -> Any:
    """Native async OpenRouter chat completion using a caller-provided client."""
    return await client.chat.completions.create(
        **_build_chat_params(messages, model, reasoning_effort, enable_reasoning, **kwargs)
    )


class LLMWrapper:
    """Wrapper for LLM calls with caching and retries."""
    
    def __init__(self, cache_dir: str, max_retries: int = 3, enable_reasoning: bool = False):
        self.cache_dir = ensure_dir(cache_dir) / "llm"
        ensure_dir(self.cache_dir)
        self.max_retries = max_retries
        self.call_count = 0
        self.cost_tracker = get_cost_tracker()
        self.current_component = None  # Will be set by calling code
        # Whether to forward reasoning effort to the model (only useful for
        # reasoning-capable models; default model is non-reasoning).
        self.enable_reasoning = enable_reasoning
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
    
    def _extract_json(self, response: str) -> Dict[str, Any]:
        """Extract JSON from response, handling various formats."""
        response = response.strip()
        
        # Try direct JSON parse first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Look for JSON within markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end > start:
                json_str = response[start:end].strip()
                return json.loads(json_str)
        
        # Look for JSON within regular code blocks
        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                json_str = response[start:end].strip()
                return json.loads(json_str)
        
        # Look for anything that looks like JSON (starts with { or [)
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith(('{', '[')):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        
        raise ValueError(f"Could not extract valid JSON from response: {response[:200]}...")
    
    def get_stats(self) -> Dict[str, int]:
        """Get usage statistics."""
        return {"total_calls": self.call_count}

    async def batch_json_complete(
        self,
        prompts: List[str],
        model: str,
        cache_keys: Optional[List[Optional[str]]] = None,
        schema_hints: Optional[List[Optional[str]]] = None,
        batch_size: int = 8,
        max_retries: int = 3,
        reasoning_effort: str = "low",
        retry_delay_base: float = 1.0,
        verbosity: int = 0,
        print_reasoning_summary: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Process multiple prompts in parallel batches with rate limit retry handling.
        
        Args:
            prompts: List of prompts to process
            model: Model name (e.g., "gpt-4o-mini")
            cache_keys: Optional list of cache keys (if None, no caching)
            schema_hints: Optional list of schema hints
            batch_size: Maximum number of concurrent requests (default: 16)
            max_retries: Maximum retries for rate limit errors (default: 3)
            retry_delay_base: Base delay for exponential backoff (default: 1.0s)
            
        Returns:
            List of parsed JSON responses in same order as input prompts
        """
        if not prompts:
            return []
        
        n_prompts = len(prompts)
        cache_keys = cache_keys or [None] * n_prompts
        schema_hints = schema_hints or [None] * n_prompts
        
        print(f"Processing {n_prompts} prompts in batches of {batch_size}")
        
        # Check cache first and prepare uncached tasks
        results = [None] * n_prompts
        uncached_indices = []
        
        for i, (prompt, cache_key, schema_hint) in enumerate(zip(prompts, cache_keys, schema_hints)):
            if cache_key:
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
            # Process uncached prompts in batches
            for batch_start in range(0, len(uncached_indices), batch_size):
                batch_end = min(batch_start + batch_size, len(uncached_indices))
                batch_indices = uncached_indices[batch_start:batch_end]

                print(f"Processing batch {batch_start//batch_size + 1}: items {batch_start + 1}-{batch_end} of {len(uncached_indices)} uncached")

                # Create tasks for this batch
                tasks = []
                for i in batch_indices:
                    prompt = prompts[i]
                    cache_key = cache_keys[i]
                    schema_hint = schema_hints[i]

                    task = asyncio.create_task(self._async_json_complete_with_retry(
                        prompt=prompt,
                        model=model,
                        cache_key=cache_key,
                        schema_hint=schema_hint,
                        max_retries=max_retries,
                        retry_delay_base=retry_delay_base,
                        reasoning_effort=reasoning_effort,
                        verbosity=verbosity,
                        print_reasoning_summary=print_reasoning_summary
                    ))
                    tasks.append(task)

                # Execute batch
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Store results, handling exceptions
                for i, result in zip(batch_indices, batch_results):
                    if isinstance(result, Exception):
                        print(f"Task failed for prompt {i}: {result}")
                        results[i] = result  # Keep exception in results for caller to handle
                    else:
                        results[i] = result
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
        reasoning_effort: str = "low",
        verbosity: int = 0,
        print_reasoning_summary: bool = False
    ) -> Dict[str, Any]:
        """Single async JSON completion with retry logic."""
        # Check cache first
        if cache_key:
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

                response = await async_chat_completion(
                    self._async_client,
                    messages=[{"role": "user", "content": json_prompt}],
                    model=model,
                    reasoning_effort=reasoning_effort,
                    enable_reasoning=self.enable_reasoning,
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
                if cache_key:
                    try:
                        cache_path = get_cache_path(self.cache_dir, cache_key)
                        save_json(result, cache_path)
                    except Exception as e:
                        print(f"Warning: Failed to save cache {cache_path}: {e}")
                
                # Update call count (thread-safe increment)
                self.call_count += 1
                
                return result
                
            except Exception as e:
                if attempt < max_retries and self._is_rate_limit_error(e):
                    delay = retry_delay_base * (2 ** attempt)
                    print(f"Rate limit error, retrying in {delay}s: {e}")
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