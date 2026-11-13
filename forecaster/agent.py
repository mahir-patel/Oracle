"""Forecasting agents: the real LLM ``Agent`` and ``FakeAgent``, the offline
stand-in used by tests and the simulation."""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Protocol

import anthropic
import numpy as np

from forecaster.aggregate import logit, sigmoid
from forecaster.config import Config
from forecaster.models import Question, Sample
from forecaster.prompts import SYSTEM_PROMPT, persona_for, user_message

PROB_LO, PROB_HI = 0.01, 0.99
MAX_WORKERS = 5
MAX_ATTEMPTS = 3  # per API request
BACKOFF_SECONDS = 2.0  # then 4.0
MAX_CONTINUATIONS = 3  # resumes of a turn the API paused

FORECAST_RE = re.compile(r"<forecast>\s*(\{.*?\})\s*</forecast>", re.DOTALL)


class ForecastAgent(Protocol):
    """What the pipeline needs from an agent. The counters are running totals."""

    model: str
    cost_usd: float
    n_searches: int
    parse_failures: int

    def forecast(self, q: Question, n: int, now: datetime) -> list[Sample]: ...


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def parse_forecast(text: str, persona: str) -> Sample | None:
    """Read the ``<forecast>{...}</forecast>`` block. None if it is missing or garbled.

    The last block wins: the prompt asks for it at the very end, and the
    reasoning before it may quote the format.
    """
    blocks = FORECAST_RE.findall(text)
    if not blocks:
        return None
    try:
        data = json.loads(blocks[-1])
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    probability = _number(data.get("probability"))
    if probability is None:
        return None
    base_rate = _number(data.get("base_rate"))
    if base_rate is not None and not 0.0 <= base_rate <= 1.0:
        base_rate = None
    return Sample(
        persona=persona,
        probability=min(max(probability, PROB_LO), PROB_HI),
        rationale=str(data.get("rationale") or "").strip(),
        base_rate=base_rate,
    )


class RetriesExhausted(Exception):
    """A request kept failing with a retryable error."""


class Agent:
    """The real forecaster: one LLM pass with web search per persona, in parallel.

    Running totals (``cost_usd``, ``n_searches``, ``parse_failures``,
    ``api_failures``) cover everything this agent has done. Once ``cost_usd``
    reaches ``config.max_spend_per_run`` no further request is started.

    Errors that a retry cannot fix (bad request, bad API key) are raised, so a
    misconfigured run stops instead of logging empty forecasts.
    """

    def __init__(
        self,
        config: Config,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        keep_transcripts: bool = False,
    ) -> None:
        self.config = config
        self.model = config.model
        # The SDK's own retries are off: the backoff here is the only one.
        self.client = client if client is not None else anthropic.Anthropic(max_retries=0)
        self.sleep = sleep
        self.cost_usd = 0.0
        self.n_searches = 0
        self.parse_failures = 0
        self.api_failures = 0
        self.input_tokens = 0
        self.output_tokens = 0
        # Full reply text of every pass, for inspection. Off by default.
        self.transcripts: list[dict[str, Any]] | None = [] if keep_transcripts else None
        self._lock = threading.Lock()

    @property
    def budget_exhausted(self) -> bool:
        return self.cost_usd >= self.config.max_spend_per_run

    def forecast(self, q: Question, n: int, now: datetime) -> list[Sample]:
        if n <= 0:
            return []
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, n)) as pool:
            results = list(pool.map(lambda i: self._one_pass(q, i, now), range(n)))
        return [s for s in results if s is not None]

    # ---------- one pass ----------

    def _tools(self) -> list[dict[str, Any]]:
        tool: dict[str, Any] = {
            "type": self.config.web_search_tool,
            "name": "web_search",
            "max_uses": self.config.max_web_searches,
        }
        blocked = [d.strip() for d in self.config.search_blocked_domains.split(",") if d.strip()]
        if blocked:
            tool["blocked_domains"] = blocked
        return [tool]

    def _one_pass(self, q: Question, index: int, now: datetime) -> Sample | None:
        persona, persona_text = persona_for(index)
        if self.budget_exhausted:
            return None
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_message(q, persona_text, now)}
        ]
        texts: list[str] = []
        stop_reason = None
        used = {"requests": 0, "input_tokens": 0, "cached_tokens": 0, "output_tokens": 0,
                "searches": 0, "cost_usd": 0.0}  # fmt: skip
        try:
            for _ in range(MAX_CONTINUATIONS + 1):
                response = self._request(messages)
                used["requests"] += 1
                for key, value in self._record_usage(response.usage).items():
                    used[key] += value
                texts += [b.text for b in response.content if b.type == "text"]
                stop_reason = response.stop_reason
                # A long search turn can be paused by the API; sending the
                # paused reply back unchanged resumes it.
                if stop_reason != "pause_turn" or self.budget_exhausted:
                    break
                messages.append({"role": "assistant", "content": response.content})
        except RetriesExhausted:
            with self._lock:
                self.api_failures += 1
            return None

        # Citations split one sentence across several text blocks, so join with nothing.
        text = "".join(texts)
        sample = parse_forecast(text, persona)
        with self._lock:
            if sample is None:
                self.parse_failures += 1
            if self.transcripts is not None:
                self.transcripts.append(
                    {
                        "market_id": q.market_id,
                        "persona": persona,
                        "stop_reason": stop_reason,
                        "parsed": sample is not None,
                        "text": text,
                        **used,
                    }
                )
        return sample

    def _request(self, messages: list[dict[str, Any]]) -> Any:
        """One ``messages.create`` call; retries rate limits, 5xx and network errors."""
        last: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                return self.client.messages.create(
                    model=self.model,
                    max_tokens=self.config.max_tokens,
                    system=SYSTEM_PROMPT,
                    tools=self._tools(),
                    messages=messages,
                )
            except (anthropic.RateLimitError, anthropic.APIConnectionError) as exc:
                last = exc
            except anthropic.APIStatusError as exc:
                if exc.status_code < 500:
                    raise
                last = exc
            if attempt < MAX_ATTEMPTS - 1:
                self.sleep(BACKOFF_SECONDS * 2**attempt)
        raise RetriesExhausted(str(last)) from last

    def _record_usage(self, usage: Any) -> dict[str, float]:
        """Tokens at the list prices plus the per-search fee. Cached input tokens,
        if any, are charged at the full input price, which can only overstate.
        Returns what this one response used."""
        cached = getattr(usage, "cache_read_input_tokens", 0) or 0
        tokens_in = (
            (usage.input_tokens or 0)
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
            + cached
        )
        tokens_out = usage.output_tokens or 0
        server_tools = getattr(usage, "server_tool_use", None)
        searches = (getattr(server_tools, "web_search_requests", 0) or 0) if server_tools else 0
        cost = (
            tokens_in * self.config.price_in / 1e6
            + tokens_out * self.config.price_out / 1e6
            + searches * self.config.price_per_search
        )
        with self._lock:
            self.input_tokens += tokens_in
            self.output_tokens += tokens_out
            self.n_searches += searches
            self.cost_usd += cost
        return {
            "input_tokens": tokens_in,
            "cached_tokens": cached,
            "output_tokens": tokens_out,
            "searches": searches,
            "cost_usd": cost,
        }


class FakeAgent:
    """A noisy, hedging forecaster with access to the true probability.

    Each sample is ``sigmoid(hedge * logit(truth) + N(0, noise))`` clipped to
    [0.01, 0.99]: ``hedge`` < 1 pulls it toward 50% the way LLMs do, and
    ``noise`` is independent per sample. It costs nothing and never fails.
    """

    model = "fake"

    def __init__(
        self,
        truth_fn: Callable[[Question], float],
        noise: float = 0.5,
        hedge: float = 0.7,
        seed: int | Sequence[int] = 0,
    ) -> None:
        self.truth_fn = truth_fn
        self.noise = noise
        self.hedge = hedge
        self.rng = np.random.default_rng(seed)
        self.cost_usd = 0.0
        self.n_searches = 0
        self.parse_failures = 0

    def forecast(self, q: Question, n: int, now: datetime) -> list[Sample]:
        centre = self.hedge * logit(self.truth_fn(q))
        samples = []
        for i in range(n):
            p = sigmoid(centre + float(self.rng.normal(0, self.noise)))
            p = min(max(p, PROB_LO), PROB_HI)
            samples.append(Sample(persona=f"fake_{i}", probability=p, rationale="simulated"))
        return samples
