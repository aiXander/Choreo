"""LLM wrapper for OpenRouter (OpenAI-compatible API) with caching and retry logic.

All LLM calls are routed through OpenRouter (https://openrouter.ai) using the
OpenAI Python SDK pointed at OpenRouter's base URL. This lets us swap providers
and run different pipeline phases with different models simply by changing the
model strings in config.yaml under `models:` (e.g. "google/gemini-3.1-flash-lite",
"anthropic/claude-3.5-sonnet", "openai/gpt-5").
"""

import os
import json
import math
import time
import random
import asyncio
import contextlib
import concurrent.futures
import email.utils
from collections import deque
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Any, Optional, List, Sequence

from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

from .utils import get_cache_path, load_json, save_json, ensure_dir
from .cost_tracker import get_cost_tracker

# Load environment variables (OPENROUTER_API_KEY) from the repo-root .env if present.
load_dotenv()

# OpenRouter exposes an OpenAI-compatible REST API.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# The default embedding model, used when `models.embedding` is unset. There is
# deliberately NO chat-model equivalent: `choreo/defaults/config.yaml` carries
# `models.extraction_llm` / `models.pair_llm`, and a transport that quietly
# substitutes a model the caller did not ask for is a bug, not a convenience —
# a host overriding one phase to `None` used to fall through to a packaged
# Gemini slug on a call carrying member profile material, silently, at a
# different price and a different provider policy. `_build_chat_params` raises
# instead.
DEFAULT_EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"

#: A model slug carrying this suffix already names a COMPLETE routing policy on
#: OpenRouter's side (an account preset: provider order, quantization floor,
#: retention posture). A request-level `provider` object REPLACES that preset
#: wholesale rather than merging with it, so a slug with this suffix must travel
#: with no `provider` block at all — see `_build_extra_body`.
PRESET_SEP = "@preset/"

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
        # fail fast into our own retry loop (`_async_json_complete_with_retry`:
        # jittered backoff, Retry-After, adaptive concurrency) instead of
        # pinning a batch at N-1/N for the SDK's 600s default × retries — and
        # an SDK-level retry would also bypass the limiter's 429 accounting.
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


def _build_extra_body(
    reasoning_effort: Optional[str],
    model: str,
    fallback_models: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Assemble OpenRouter-specific request extensions.

    Note: OpenRouter usage accounting (cost + token details) is always on, so no
    ``usage: {include: true}`` flag is needed — it returns automatically.

    **Routing is either a preset or a provider block, never both.** Member
    profile material reaches every Choreo LLM phase, so a route whose provider
    policy permits data collection must be impossible. There are two ways to say
    that and they are mutually exclusive:

    * the slug carries ``@preset/<name>`` — the host has named a complete
      routing policy on its OpenRouter account (provider order, quantization
      floor, retention). Sending a ``provider`` object alongside it REPLACES the
      preset wholesale while every request keeps returning 200, so this emits
      none;
    * the slug is bare (standalone Choreo, no host account) — emit the
      ``data_collection: deny`` floor, as this has always done.
    """
    extra_body: Dict[str, Any] = {}
    if PRESET_SEP not in (model or ""):
        extra_body["provider"] = {"data_collection": "deny"}
    # A prioritized chain: OpenRouter fails over INSIDE one request, so a
    # provider outage costs one degraded answer rather than a failed phase.
    chain = [m for m in (fallback_models or []) if m]
    if len(chain) > 1:
        extra_body["models"] = list(chain)
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
    fallback_models: Optional[Sequence[str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Build the chat.completions request params shared by sync and async paths.

    Raises on a missing model rather than substituting one: see the note on
    ``DEFAULT_EMBEDDING_MODEL`` for why there is no chat default any more.
    """
    if not model:
        raise ValueError(
            "no model for this phase — set models.extraction_llm / models.pair_llm "
            "in config.yaml (or pass one explicitly); this transport substitutes none"
        )
    params: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }

    extra_body = _build_extra_body(reasoning_effort, model, fallback_models)
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
    fallback_models: Optional[Sequence[str]] = None,
    **kwargs,
) -> Any:
    """Native async OpenRouter chat completion using a caller-provided client."""
    return await client.chat.completions.create(
        **_build_chat_params(messages, model, reasoning_effort, fallback_models, **kwargs)
    )


class JSONExtractionError(ValueError):
    """Raised when a model response cannot be parsed into JSON.

    Treated as retryable (see ``_is_retryable_error``): LLM output is stochastic,
    so re-sampling the same prompt usually yields valid JSON on the next attempt.
    """


# ---------------------------------------------------------------------------
# Response-shape validation
#
# `response_format={"type":"json_object"}` (json_mode) guarantees SYNTAX, never
# SCHEMA: the provider promises *an* object, not *your* keys. A phase that reads
# its fields with `dict.get(key, <default>)` therefore cannot tell a wrong-shaped
# answer from a real one — it silently ships the default. That is exactly how a
# 2026-09-02 wintercircus match run published two cards carrying placeholder
# prose: one response came back as `{"": "{\"intro_for_a\": …}"}` (the real
# answer, stringified inside an empty-key wrapper) and one as `{": ": ", "}`
# (degenerate). Both parsed fine, both were cached, neither was retried.
#
# A phase opts in by declaring `required_key_sets`; every phase that declares
# nothing keeps the previous behavior byte-for-byte.
# ---------------------------------------------------------------------------

# Alternative key groups a response may satisfy. `None`/empty = no requirement.
KeySets = Optional[Sequence[Sequence[str]]]

# How deep / how many nodes the envelope unwrap will look through. Real
# responses are a handful of keys, so these only bound a pathological payload.
_MAX_ENVELOPE_DEPTH = 3
_MAX_ENVELOPE_NODES = 16


def _is_filled(value: Any) -> bool:
    """Whether a decoded JSON value counts as present (not null/blank/empty)."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True  # numbers/bools are legitimate answers, including 0 and False


def satisfies_key_sets(value: Any, key_sets: KeySets) -> bool:
    """Whether ``value`` is a JSON object carrying one COMPLETE required key set.

    ``key_sets`` lists ALTERNATIVE groups (e.g. a directional format and a legacy
    one): the value passes if every key of any ONE group is present and filled.
    With no key sets declared everything passes — the opt-out that keeps
    undeclared phases unchanged.
    """
    if not key_sets:
        return True
    if not isinstance(value, dict):
        return False
    return any(
        keys and all(_is_filled(value.get(key)) for key in keys)
        for keys in key_sets
    )


def unwrap_json_envelope(value: Any, key_sets: KeySets) -> Any:
    """Recover an answer a model stringified inside a wrapper object.

    Deliberately conservative: it returns something other than ``value`` ONLY
    when ``value`` fails ``key_sets`` and a decoded inner object fully satisfies
    them. So it can never rewrite a well-formed response, never turns a scalar
    string value into the answer, and is a no-op when no key sets are declared.
    """
    if not key_sets or satisfies_key_sets(value, key_sets):
        return value
    frontier = [value]
    budget = _MAX_ENVELOPE_NODES
    for _ in range(_MAX_ENVELOPE_DEPTH):
        next_frontier: List[Any] = []
        for node in frontier:
            if not isinstance(node, dict):
                continue
            for raw in node.values():
                if not isinstance(raw, str) or not raw.strip():
                    continue
                if budget <= 0:
                    return value
                budget -= 1
                try:
                    decoded = json.loads(raw, strict=False)
                except (ValueError, TypeError):
                    continue
                if satisfies_key_sets(decoded, key_sets):
                    return decoded
                if isinstance(decoded, dict):
                    next_frontier.append(decoded)
        if not next_frontier:
            break
        frontier = next_frontier
    return value


def _describe_shape(value: Any) -> str:
    """Short, log-safe description of a response's shape (never its content)."""
    if isinstance(value, dict):
        return f"object with keys {sorted(map(str, value))[:8]}"
    return type(value).__name__


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


# ---------------------------------------------------------------------------
# Retry backoff, adaptive concurrency, per-phase stats
#
# The old loop was `base * 2**attempt` (1 s, 2 s, 4 s), no jitter, no
# Retry-After, 3 retries. Fine at 16 in flight; at 128 (the first whole-
# community wintercircus run: ~3k scoring + ~3k intro calls) one upstream 429
# burst made every in-flight call retry in LOCKSTEP and give up after ~7 s —
# and a give-up is silent downstream (a scoring group falls back to embed-only
# ranks, an intro to the honest-but-generic fallback card). So: full jitter
# with a cap, Retry-After honored, more retries, and AIMD on the in-flight
# limit so a 429 storm sheds load instead of re-hammering. Deadline-bounded
# batches (the live `find_matches` re-rank) keep the legacy policy exactly —
# see `batch_json_complete`.
# ---------------------------------------------------------------------------

#: Default retries for an unbounded (batch) phase. With full jitter capped at
#: 30 s the worst-case sleep budget is 1+2+4+8+16+30 ≈ 61 s (mean half that) —
#: long enough to ride out a provider's rate-limit window, short enough that a
#: genuinely dead route still fails inside a couple of minutes.
BATCH_MAX_RETRIES = 6
#: Retries for a deadline-bounded batch when the caller names none: the
#: pre-AIMD default, kept so the live query re-rank path is byte-for-byte the
#: same retry count and sleeps (`1 s, 2 s, 4 s`, unjittered) as before.
LEGACY_MAX_RETRIES = 3
#: Upper bound on any single backoff sleep, including a server's Retry-After.
RETRY_DELAY_CAP_S = 30.0
#: AIMD knobs. Halve on a rate limit (never below the floor — 8 keeps a phase
#: moving even under sustained 429s; a smaller ceiling wins), +1 per 8
#: consecutive successes back to the configured ceiling. 8 rather than 1:
#: recovering 64→128 then takes ~500 calls, so a limit we just tripped isn't
#: re-tripped a few seconds later.
ADAPTIVE_FLOOR = 8
ADAPTIVE_DECREASE_FACTOR = 0.5
ADAPTIVE_INCREASE_EVERY = 8

# A Retry-After value longer than this is not a header, it's garbage.
_MAX_RETRY_HEADER_LEN = 64
# Module RNG so tests can swap in a seeded/stub one without touching `random`.
_RETRY_RNG = random.Random()


def _read_header(headers: Any, name: str) -> Optional[str]:
    """One header value as a short string, or None — never raises."""
    try:
        value = headers.get(name)
        if value is None and isinstance(headers, dict):
            # Plain dicts aren't case-insensitive like httpx.Headers.
            value = next((v for k, v in headers.items()
                          if isinstance(k, str) and k.lower() == name), None)
    except Exception:  # pylint: disable=broad-except
        return None
    if isinstance(value, bytes):
        value = value.decode("latin-1", errors="replace")
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > _MAX_RETRY_HEADER_LEN:
        return None
    return value


def _finite_non_negative(raw: str) -> Optional[float]:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value


def parse_retry_after(error: BaseException, now: Optional[float] = None) -> Optional[float]:
    """Seconds the server asked us to wait, from an SDK error's response headers.

    Reads ``retry-after-ms`` (OpenAI-style, milliseconds) first, then
    ``retry-after`` as delta-seconds or an HTTP-date. Anything absent,
    oversized, negative, non-finite or unparseable is ``None`` ("no advice"),
    and the caller clamps whatever comes back to ``RETRY_DELAY_CAP_S`` — a
    server's say-so must not park a batch for an hour.
    """
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw_ms = _read_header(headers, "retry-after-ms")
    if raw_ms is not None:
        ms = _finite_non_negative(raw_ms)
        if ms is not None:
            return ms / 1000.0
    raw = _read_header(headers, "retry-after")
    if raw is None:
        return None
    seconds = _finite_non_negative(raw)
    if seconds is not None:
        return seconds
    try:
        when = email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if when is None or when.tzinfo is None:
        return None
    delta = when.timestamp() - (time.time() if now is None else now)
    return max(0.0, delta) if math.isfinite(delta) else None


def compute_retry_delay(
    attempt: int,
    kind: str,
    *,
    base: float,
    cap: float = RETRY_DELAY_CAP_S,
    retry_after: Optional[float] = None,
    rng: Optional[random.Random] = None,
) -> float:
    """Sleep before retry ``attempt`` (0-based) for an error of ``kind``.

    * ``"sample"`` — malformed JSON / failed shape gate / JSON mode rejected.
      Re-sampling is the fix and the provider isn't pushing back, so a short
      jittered pause (``≤ base``) that does NOT grow with the attempt number.
    * ``"rate_limit"`` / ``"transient"`` — full jitter,
      ``uniform(0, min(cap, base·2^attempt))``: spreads a burst of simultaneous
      failures across the whole window instead of re-firing them together.
      With a server ``retry_after`` the wait is that, plus the same jitter so
      128 calls told "5 s" don't all return at t+5 s — clamped to ``cap``.
    """
    rng = rng or _RETRY_RNG
    cap = max(0.0, float(cap))
    base = max(0.0, float(base))
    if kind == "sample":
        return rng.uniform(0.0, min(cap, base))
    spread = min(cap, base * (2 ** min(max(attempt, 0), 30)))
    if retry_after is not None:
        return min(cap, max(0.0, retry_after) + rng.uniform(0.0, spread))
    return rng.uniform(0.0, spread)


class AdaptiveConcurrencyLimiter:
    """An AIMD in-flight limit: a semaphore whose size can shrink and regrow.

    ``asyncio.Semaphore`` can't shrink, so this keeps its own count and a FIFO
    of waiters, and HANDS a permit to a waiter on release (the waiter never
    re-checks), which makes it leak-free under cancellation: a waiter cancelled
    before it got a permit just leaves the queue; one cancelled after it was
    handed one gives it straight back. Shrinking withholds permits — calls
    already in flight finish, nothing new starts until ``in_flight < limit``.

    Rate-limit cuts are de-duplicated by ``epoch``: a caller passes the epoch
    it saw when its request STARTED, so the 127 other 429s from the same burst
    don't halve the limit 127 times — only a 429 from a request dispatched
    after the latest cut can cut again.

    With ``adaptive=False`` it is a plain fixed-size limiter (what a
    deadline-bounded batch uses).
    """

    def __init__(
        self,
        ceiling: int,
        *,
        floor: int = ADAPTIVE_FLOOR,
        decrease_factor: float = ADAPTIVE_DECREASE_FACTOR,
        increase_every: int = ADAPTIVE_INCREASE_EVERY,
        adaptive: bool = True,
        label: str = "",
    ):
        self.ceiling = max(1, int(ceiling))
        self.floor = max(1, min(int(floor), self.ceiling))
        self.decrease_factor = float(decrease_factor)
        self.increase_every = max(1, int(increase_every))
        self.adaptive = adaptive
        self.label = label
        self.limit = self.ceiling
        self.min_limit = self.ceiling
        self.in_flight = 0
        self.peak_in_flight = 0
        self.epoch = 0
        self._successes = 0
        self._waiters: "deque[asyncio.Future]" = deque()

    async def acquire(self, priority: bool = False) -> None:
        """Take a permit. ``priority`` jumps the queue (used by retries, so a
        phase's tail isn't every retried call queued behind all fresh work)."""
        if self.in_flight < self.limit and not self._waiters:
            self._take()
            return
        fut = asyncio.get_running_loop().create_future()
        if priority:
            self._waiters.appendleft(fut)
        else:
            self._waiters.append(fut)
        try:
            await fut
        except BaseException:
            if fut.done() and not fut.cancelled():
                self.release()  # handed a permit, then cancelled: give it back
            else:
                with contextlib.suppress(ValueError):
                    self._waiters.remove(fut)
            raise

    def release(self) -> None:
        if self.in_flight <= 0:
            raise ValueError("AdaptiveConcurrencyLimiter released more than acquired")
        self.in_flight -= 1
        self._grant()

    @contextlib.asynccontextmanager
    async def slot(self, priority: bool = False):
        await self.acquire(priority)
        try:
            yield
        finally:
            self.release()

    def on_rate_limit(self, epoch: Optional[int] = None) -> bool:
        """Multiplicative decrease. Returns whether the limit actually moved."""
        self._successes = 0
        if not self.adaptive or (epoch is not None and epoch != self.epoch):
            return False
        new = max(self.floor, int(self.limit * self.decrease_factor))
        if new >= self.limit:
            return False
        old, self.limit = self.limit, new
        self.epoch += 1
        self.min_limit = min(self.min_limit, new)
        print(f"{self.label}LLM concurrency {old} → {new} "
              f"(rate limited; ceiling {self.ceiling}, floor {self.floor})")
        return True

    def on_success(self) -> None:
        """Additive increase: +1 per ``increase_every`` consecutive successes."""
        if not self.adaptive or self.limit >= self.ceiling:
            return
        self._successes += 1
        if self._successes < self.increase_every:
            return
        self._successes = 0
        self.limit += 1
        if self.limit >= self.ceiling:
            # Sparse on purpose: cuts and full recovery only — the progress
            # ticker shows the in-between value while the limit is below ceiling.
            print(f"{self.label}LLM concurrency recovered to {self.ceiling}")
        self._grant()

    def _take(self) -> None:
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)

    def _grant(self) -> None:
        while self._waiters and self.in_flight < self.limit:
            fut = self._waiters.popleft()
            if fut.done():  # cancelled while queued
                continue
            self._take()
            fut.set_result(None)


@dataclass
class BatchStats:
    """What one ``batch_json_complete`` phase did — so silent degradation has
    a count. Every field is a plain int/float, ready for a host run summary.

    * ``prompts`` — prompts handed in; ``cache_hits`` — served from the file
      cache without a call.
    * ``dispatched`` — prompts that made at least one API attempt.
    * ``retries`` — extra attempts (all causes); ``rate_limited`` — responses
      classified as a rate limit (429), including a final one.
    * ``failed`` — prompts whose slot holds an Exception (retries exhausted or
      non-retryable); ``cancelled`` — slots left ``None`` at ``deadline_s``.
    * ``ceiling`` / ``min_concurrency`` — configured in-flight cap and the
      lowest effective limit AIMD reached (== ceiling when never cut).
    """

    label: Optional[str] = None
    prompts: int = 0
    cache_hits: int = 0
    dispatched: int = 0
    retries: int = 0
    rate_limited: int = 0
    failed: int = 0
    cancelled: int = 0
    ceiling: int = 0
    min_concurrency: int = 0
    elapsed_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary_line(self) -> str:
        label = f"[{self.label}] " if self.label else ""
        line = (
            f"{label}LLM phase: {self.prompts} prompts · {self.cache_hits} cached · "
            f"{self.dispatched} called · {self.retries} retries "
            f"({self.rate_limited} rate-limited) · {self.failed} failed"
        )
        if self.cancelled:
            line += f" · {self.cancelled} cancelled at deadline"
        if self.ceiling:
            line += f" · min concurrency {self.min_concurrency}/{self.ceiling}"
        return line + f" · {self.elapsed_s:.1f}s"


class LLMWrapper:
    """Wrapper for LLM calls with caching and retries."""
    
    def __init__(self, cache_dir: Optional[str] = None, max_retries: int = BATCH_MAX_RETRIES, reasoning_effort: Optional[str] = "low",
                 max_concurrent_llm_calls: int = 16,
                 fallback_models: Optional[Sequence[str]] = None,
                 checkpoint: Optional[Callable[[], None]] = None):
        # cache_dir=None disables the file-based response cache entirely
        # (in-memory pipeline runs, e.g. transient query matching).
        if cache_dir:
            self.cache_dir = ensure_dir(cache_dir) / "llm"
            ensure_dir(self.cache_dir)
        else:
            self.cache_dir = None
        # Retries for an UNBOUNDED batch phase that names none per call (was
        # dead config before the AIMD rework). A deadline-bounded batch ignores
        # it and keeps LEGACY_MAX_RETRIES — see batch_json_complete.
        self.max_retries = max_retries
        # One BatchStats per batch_json_complete call, in order; get_stats()
        # sums them so a host can put retries/failures in its run summary.
        self.phase_stats: List[BatchStats] = []
        self.call_count = 0
        # Number of new responses written to the file cache this run. Lets an
        # adapter skip persistence work (e.g. Modal volume.commit) when a call
        # was served entirely from cache and nothing on disk changed.
        self.cache_writes = 0
        # Optional host hook to make the cache durable mid-run (e.g. a Modal
        # volume.commit), called off the event loop at the end of every
        # batch_json_complete that wrote >=1 new entry — i.e. between phases
        # (scoring, intros, HyDE, extraction), never mid-phase. A long run that
        # dies after scoring then retries with its scores already cached. A
        # failing checkpoint is logged and swallowed: durability is a bonus,
        # never a reason to fail the run.
        self.checkpoint = checkpoint
        self.cost_tracker = get_cost_tracker()
        self.current_component = None  # Will be set by calling code
        # Default reasoning effort forwarded to OpenRouter (xhigh|high|medium|
        # low|minimal|none, or None to let the model decide). Used by any phase
        # that doesn't pass an explicit reasoning_effort; "low" is the
        # cost-effective baseline and is ignored on non-reasoning models.
        self.reasoning_effort = reasoning_effort
        # The prioritized model chain for every chat phase, primary FIRST, sent
        # as OpenRouter's `models` array so failover happens inside one request.
        # None (the standalone default) means no chain: one model, one attempt
        # per retry. A host sets this from its own model contract; the entries
        # must already be whatever routed form the host uses, because this class
        # never rewrites a slug.
        self.fallback_models = list(fallback_models) if fallback_models else None
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
    
    def _load_cached(self, cache_key: Optional[str], required_key_sets: KeySets = None) -> Any:
        """Return a usable cached response, or ``None`` for "regenerate".

        Three misses, all silent-safe:

        * no key / no cache dir / no file — an ordinary miss;
        * an unreadable file — warned, then regenerated (pre-existing behavior);
        * a file that does not satisfy ``required_key_sets`` — **the poisoned
          entry**. Responses are cached before the caller ever looks at their
          shape, so one bad generation used to pin itself into the cache under a
          content-addressed key and replay on every later run (the prompt hash
          only changes when the profile text does). Treating it as a miss makes
          the cache self-healing instead. The envelope unwrap runs first, so a
          stringified-but-complete cached answer is recovered without re-spending.
        """
        if not (cache_key and self.cache_dir):
            return None
        cache_path = get_cache_path(self.cache_dir, cache_key)
        if not cache_path.exists():
            return None
        try:
            cached = load_json(cache_path)
        except Exception as e:
            print(f"Warning: Failed to load cache {cache_path}: {e}")
            return None
        cached = unwrap_json_envelope(cached, required_key_sets)
        if not satisfies_key_sets(cached, required_key_sets):
            print(
                f"⚠️  Ignoring unusable cached response {cache_path.name} "
                f"({_describe_shape(cached)}); regenerating"
            )
            return None
        return cached

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
    
    def get_stats(self) -> Dict[str, Any]:
        """Cumulative usage across every batch phase this wrapper ran.

        ``total_calls`` is successful API calls (unchanged meaning). The rest
        sum ``BatchStats`` over ``phase_stats`` (see that class for each
        field); ``min_concurrency`` is the lowest AIMD limit any phase reached,
        ``None`` before the first phase. Per-phase detail: ``get_phase_stats``.
        """
        totals: Dict[str, Any] = {"total_calls": self.call_count, "phases": len(self.phase_stats)}
        for key in ("prompts", "cache_hits", "dispatched", "retries",
                    "rate_limited", "failed", "cancelled"):
            totals[key] = sum(getattr(s, key) for s in self.phase_stats)
        floors = [s.min_concurrency for s in self.phase_stats if s.ceiling]
        totals["min_concurrency"] = min(floors) if floors else None
        return totals

    def get_phase_stats(self) -> List[Dict[str, Any]]:
        """One dict per batch_json_complete call (label = its progress_label)."""
        return [s.to_dict() for s in self.phase_stats]

    async def batch_json_complete(
        self,
        prompts: List[str],
        model: str,
        cache_keys: Optional[List[Optional[str]]] = None,
        schema_hints: Optional[List[Optional[str]]] = None,
        max_concurrent: Optional[int] = None,
        max_retries: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        retry_delay_base: float = 1.0,
        verbosity: int = 0,
        print_reasoning_summary: bool = False,
        max_tokens: Optional[int] = None,
        progress_label: Optional[str] = None,
        progress_interval: float = 10.0,
        deadline_s: Optional[float] = None,
        required_key_sets: KeySets = None,
        retry_delay_cap: float = RETRY_DELAY_CAP_S,
        adaptive_concurrency: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple prompts concurrently with rate-limit retry handling.

        Dispatch is limiter-gated: at most the current limit of HTTP requests
        are in flight at any instant, and the next prompt fires the moment one
        returns — no fire-a-batch-then-wait barrier — so the slowest call in a
        notional batch never stalls the others. A permit covers one HTTP
        attempt, not a retry's backoff sleep, so the limit bounds real requests.

        **Two retry policies, chosen by ``deadline_s``.** An unbounded phase
        (cohort/batch scoring, intros, HyDE, extraction) gets the hardened
        policy: ``self.max_retries`` (default ``BATCH_MAX_RETRIES``) retries,
        full-jitter backoff capped at ``retry_delay_cap``, server Retry-After
        honored, and AIMD on the in-flight limit (``max_concurrent`` is the
        ceiling). A deadline-bounded phase (the live ``find_matches`` re-rank)
        keeps the legacy policy unchanged — ``LEGACY_MAX_RETRIES``, unjittered
        ``retry_delay_base * 2**attempt``, no Retry-After, fixed limit — because
        there the deadline IS the retry budget: a longer server-dictated sleep
        only converts a retry into a cancellation. An explicit ``max_retries``
        always wins on either path.

        Args:
            prompts: List of prompts to process
            model: Model name (e.g., "gpt-4o-mini")
            cache_keys: Optional list of cache keys (if None, no caching)
            schema_hints: Optional list of schema hints
            max_concurrent: Max concurrent in-flight requests. None inherits the
                wrapper's ``max_concurrent_llm_calls`` (config `concurrency:`).
                The AIMD ceiling on an unbounded phase.
            max_retries: Retries per prompt for transient errors. None = the
                policy default above (wrapper ``max_retries`` unbounded,
                ``LEGACY_MAX_RETRIES`` with a deadline); an int is used as-is.
            retry_delay_base: Base delay for exponential backoff (default: 1.0s)
            retry_delay_cap: Max single backoff sleep on the hardened policy.
            adaptive_concurrency: Force AIMD on/off. None = on iff no deadline.
            deadline_s: Optional wall-clock budget for the whole batch. None
                (default) waits for every call. When set, calls still in flight
                at the deadline are cancelled and their slots stay ``None``, so
                the caller sees "no answer" for those prompts and decides what
                that means. Use it when the model's TAIL latency costs more than
                the last few answers are worth — but note cancelled calls still
                burn the tokens the provider already generated.
            required_key_sets: Alternative key groups a response must satisfy
                (see ``satisfies_key_sets``). A response matching none of them
                is a retryable ``JSONExtractionError`` and is never cached, and
                an already-cached response that no longer matches is treated as
                a cache miss. ``None`` (default) imposes no requirement.

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

        # Retry policy (see docstring): the deadline-bounded path keeps the
        # legacy count + unjittered sleeps unless the caller names a count.
        legacy_backoff = deadline_s is not None
        if max_retries is None:
            max_retries = LEGACY_MAX_RETRIES if legacy_backoff else self.max_retries
        max_retries = max(0, int(max_retries))
        if adaptive_concurrency is None:
            adaptive_concurrency = not legacy_backoff
        label = f"[{progress_label}] " if progress_label else ""
        stats = BatchStats(label=progress_label, prompts=len(prompts),
                           ceiling=max_concurrent, min_concurrency=max_concurrent)
        started_at = asyncio.get_running_loop().time()

        print(f"Processing {n_prompts} prompts with up to {max_concurrent} concurrent LLM calls")

        # Check cache first and prepare uncached tasks
        results = [None] * n_prompts
        uncached_indices = []
        
        for i, (prompt, cache_key, schema_hint) in enumerate(zip(prompts, cache_keys, schema_hints)):
            cached = self._load_cached(cache_key, required_key_sets)
            if cached is not None:
                results[i] = cached
                continue
            uncached_indices.append(i)
        
        cached_count = n_prompts - len(uncached_indices)
        stats.cache_hits = cached_count
        if cached_count > 0:
            print(f"Found {cached_count} cached results")

        if not uncached_indices:
            self._finish_phase(stats, started_at)
            return results

        writes_before = self.cache_writes
        # Open one async client for this run (bound to the current event loop)
        # and close it when done — see make_async_openrouter_client.
        self._async_client = make_async_openrouter_client()
        try:
            # Limiter-gated continuous dispatch: every uncached prompt becomes
            # a task immediately, but the limiter lets only its current limit
            # actually hit the API at once. As soon as any call returns it
            # releases its permit and the next queued task gets it — so the
            # provider is kept saturated at the limit without the slowest call
            # in a window stalling the rest (the old fire-a-batch-then-await
            # behavior). The permit is taken per HTTP attempt inside
            # _async_json_complete_with_retry, and AIMD moves the limit
            # between ADAPTIVE_FLOOR and `max_concurrent` on 429s.
            limiter = AdaptiveConcurrencyLimiter(
                max_concurrent, adaptive=adaptive_concurrency, label=label,
            )
            total = len(uncached_indices)
            completed = 0

            async def _run_one(i: int):
                nonlocal completed
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
                        required_key_sets=required_key_sets,
                        retry_delay_cap=retry_delay_cap,
                        legacy_backoff=legacy_backoff,
                        limiter=limiter,
                        stats=stats,
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
            # and is cancelled the moment all tasks finish. While AIMD holds the
            # limit below its ceiling the line says so — that's the in-between
            # signal the limiter itself deliberately doesn't log.
            async def _progress_ticker():
                try:
                    while True:
                        await asyncio.sleep(progress_interval)
                        if completed >= total:
                            break
                        pct = 100.0 * completed / total if total else 100.0
                        throttled = (f" · concurrency {limiter.limit}/{limiter.ceiling}"
                                     if limiter.limit < limiter.ceiling else "")
                        print(f"{label}LLM progress: {completed}/{total} ({pct:.0f}%){throttled}")
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
                    if isinstance(result, Exception):
                        stats.failed += 1
                if pending:
                    stats.cancelled = len(pending)
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
                stats.min_concurrency = limiter.min_limit
        finally:
            await self._async_client.close()
            self._async_client = None

        print(f"Completed processing all {n_prompts} prompts")
        self._finish_phase(stats, started_at)
        await self._checkpoint_if_written(writes_before)
        await cleanup_background_tasks()
        return results

    async def _checkpoint_if_written(self, writes_before: int) -> None:
        """Run the host `checkpoint` hook iff this batch wrote new cache entries.

        Called once every task of the batch has settled, so no cache file of
        this batch is mid-write. Off the event loop (`asyncio.to_thread`): a
        host's commit is blocking I/O and must not stall another coroutine.
        """
        if self.checkpoint is None or self.cache_writes <= writes_before:
            return
        try:
            await asyncio.to_thread(self.checkpoint)
        except Exception as e:  # never fail a run because persistence failed
            print(f"⚠️  WARNING: LLM cache checkpoint failed (run continues): {e!r}")
    
    def _finish_phase(self, stats: BatchStats, started_at: float) -> None:
        """Close out one batch phase: record its stats and print ONE line."""
        stats.elapsed_s = asyncio.get_running_loop().time() - started_at
        self.phase_stats.append(stats)
        print(stats.summary_line())

    async def _async_json_complete_with_retry(
        self,
        prompt: str,
        model: str,
        cache_key: Optional[str] = None,
        schema_hint: Optional[str] = None,
        max_retries: int = LEGACY_MAX_RETRIES,
        retry_delay_base: float = 1.0,
        reasoning_effort: Optional[str] = None,
        verbosity: int = 0,
        print_reasoning_summary: bool = False,
        max_tokens: Optional[int] = None,
        required_key_sets: KeySets = None,
        retry_delay_cap: float = RETRY_DELAY_CAP_S,
        legacy_backoff: bool = False,
        limiter: Optional[AdaptiveConcurrencyLimiter] = None,
        stats: Optional[BatchStats] = None,
    ) -> Dict[str, Any]:
        """Single async JSON completion with retry logic.

        Each HTTP attempt holds one ``limiter`` permit; the backoff sleep holds
        none, so a call waiting out a 429 doesn't occupy a slot and a retry
        re-queues at the FRONT (``priority``) behind the current limit. A
        cancellation (deadline) anywhere in here leaves no permit behind — the
        slot context manager and the limiter's own acquire both clean up.
        """
        # Check cache first
        cached = self._load_cached(cache_key, required_key_sets)
        if cached is not None:
            return cached

        json_prompt = self._prepare_json_prompt(prompt, schema_hint)
        if limiter is None:  # direct call outside a batch: unlimited, fixed
            limiter = AdaptiveConcurrencyLimiter(1 << 30, adaptive=False)
        if stats is None:
            stats = BatchStats()
        dispatched = False

        for attempt in range(max_retries + 1):
            epoch: Optional[int] = None
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

                async with limiter.slot(priority=attempt > 0):
                    if not dispatched:
                        dispatched = True
                        stats.dispatched += 1
                    # The limit generation this request was sent under — a 429
                    # from it may cut the limit only if no cut happened since.
                    epoch = limiter.epoch
                    response = await async_chat_completion(
                        self._async_client,
                        messages=[{"role": "user", "content": json_prompt}],
                        model=model,
                        reasoning_effort=reasoning_effort if reasoning_effort is not None else self.reasoning_effort,
                        fallback_models=self.fallback_models,
                        **call_kwargs,
                    )
                # The provider accepted the request — AIMD's additive half,
                # whatever the body turns out to contain.
                limiter.on_success()

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

                # Shape gate. json_mode buys syntax, not schema — so a phase
                # that declared what it needs gets a retryable failure here
                # rather than a plausible-looking object its `.get()` calls
                # quietly paper over. Runs BEFORE the cache write on purpose:
                # a rejected response must not be persisted.
                result = unwrap_json_envelope(result, required_key_sets)
                if not satisfies_key_sets(result, required_key_sets):
                    raise JSONExtractionError(
                        "Model returned JSON missing every required key set "
                        f"{[list(keys) for keys in (required_key_sets or [])]}; "
                        f"got {_describe_shape(result)}"
                    )

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
                    kind: Optional[str] = "sample"
                else:
                    kind = self._retry_kind(e)

                retry_after = None
                if kind == "rate_limit":
                    stats.rate_limited += 1
                    if epoch is not None:
                        limiter.on_rate_limit(epoch)
                if kind in ("rate_limit", "transient") and getattr(e, "status_code", None) in (429, 503):
                    retry_after = parse_retry_after(e)

                if attempt < max_retries and kind is not None:
                    stats.retries += 1
                    if legacy_backoff:
                        # Deadline-bounded path: the pre-AIMD schedule, exactly.
                        delay = retry_delay_base * (2 ** attempt)
                    else:
                        delay = compute_retry_delay(
                            attempt, kind, base=retry_delay_base,
                            cap=retry_delay_cap, retry_after=retry_after,
                        )
                    print(f"Transient LLM error ({kind}), retrying in {delay:.2f}s: {str(e)[:300]}")
                    await asyncio.sleep(delay)
                else:
                    print(f"Failed after {attempt + 1} attempts: {e}")
                    raise
        
        # Should not reach here
        raise RuntimeError("Unexpected end of retry loop")
    
    def _retry_kind(self, error: Exception) -> Optional[str]:
        """``"sample"`` | ``"rate_limit"`` | ``"transient"``, or None = don't retry.

        JSON/shape failures are checked first: their message embeds up to 200
        chars of model output, which may well contain "429" or "rate limit"
        and must not trip the limiter.
        """
        if isinstance(error, JSONExtractionError):
            return "sample"
        if not self._is_retryable_error(error):
            return None
        if getattr(error, "status_code", None) == 429 or self._is_rate_limit_error(error):
            return "rate_limit"
        return "transient"

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
