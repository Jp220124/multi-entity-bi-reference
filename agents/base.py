"""Abstract Agent interface.

Every agent in the system implements the same protocol:

    input → validated → LLM call → validated output → context write

The base class handles the common scaffolding (logging, error handling,
cost estimation, audit-log write) so each concrete agent only has to
define its prompt, its input/output schemas, and its model tier.

Adding a new agent is a single file. The orchestrator, context store,
and existing agents do not change.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, TypeVar

from anthropic import Anthropic
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput", bound=BaseModel)


class ModelTier(str, Enum):
    """Logical model tier.

    The actual model identifier is resolved at construction time from
    environment variables so the agent code remains stable across
    Claude model upgrades.
    """

    HAIKU = "haiku"
    SONNET = "sonnet"
    OPUS = "opus"


# Approximate per-million-token rates in USD. Kept in one place so cost
# estimators stay in sync. Source: Anthropic public pricing at time of
# writing. Override with real billing data in production.
_TIER_RATES_USD_PER_MILLION: dict[ModelTier, tuple[float, float]] = {
    # (input_rate, output_rate)
    ModelTier.HAIKU: (0.25, 1.25),
    ModelTier.SONNET: (3.00, 15.00),
    ModelTier.OPUS: (15.00, 75.00),
}


@dataclass(frozen=True, slots=True)
class AgentResult(Generic[TOutput]):
    """The result of a single agent run.

    Kept separate from the Pydantic output model so the orchestrator
    can reason about latency, cost, and token usage independently of
    the business payload.
    """

    output: TOutput
    latency_ms: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    model_id: str


class Agent(ABC, Generic[TInput, TOutput]):
    """Abstract base class for a single agent.

    Concrete agents override:
        - ``tier``            model tier (Haiku / Sonnet / Opus)
        - ``output_schema``   Pydantic class to validate LLM output
        - ``build_prompt``    turn validated input into a prompt string
        - ``system_prompt``   (optional) fixed system-prompt text
    """

    # Subclasses must set these.
    tier: ModelTier
    output_schema: type[TOutput]

    # Optional overrides.
    system_prompt: str = ""
    max_output_tokens: int = 2048

    # Only L3 uses extended thinking; all other agents leave this alone.
    extended_thinking_budget_tokens: int | None = None

    def __init__(self, client: Anthropic, model_id: str) -> None:
        self._client = client
        self._model_id = model_id

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, payload: TInput) -> AgentResult[TOutput]:
        """Execute the agent against a validated input.

        Raises:
            ValidationError: The LLM returned something that cannot be
                parsed into ``output_schema``. The orchestrator is
                expected to log the raw response and move on — bad LLM
                output must never reach the context layer.
        """
        prompt = self.build_prompt(payload)
        start = time.perf_counter()

        request_kwargs: dict[str, Any] = {
            "model": self._model_id,
            "max_tokens": self.max_output_tokens,
            "system": self.system_prompt or "You produce strictly valid JSON.",
            "messages": [{"role": "user", "content": prompt}],
        }

        if self.extended_thinking_budget_tokens:
            # Opus L3 only: allow the model to think for longer before
            # committing to an answer. The budget cap is there to keep
            # cost bounded and predictable.
            request_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.extended_thinking_budget_tokens,
            }

        response = self._client.messages.create(**request_kwargs)
        latency_ms = int((time.perf_counter() - start) * 1000)

        raw_text = self._extract_text(response)
        output = self._parse_and_validate(raw_text)

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = self._estimate_cost(input_tokens, output_tokens)

        logger.info(
            "agent=%s tier=%s model=%s latency_ms=%d in=%d out=%d cost_usd=%.5f",
            type(self).__name__,
            self.tier.value,
            self._model_id,
            latency_ms,
            input_tokens,
            output_tokens,
            cost,
        )

        return AgentResult(
            output=output,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
            model_id=self._model_id,
        )

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def build_prompt(self, payload: TInput) -> str:
        """Return the user-message prompt string for this input."""

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract the plain-text content from a Claude response.

        Responses with extended thinking return a list of blocks; we
        ignore the ``thinking`` block and return the concatenated
        ``text`` blocks.
        """
        parts: list[str] = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                parts.append(block.text)
        return "\n".join(parts).strip()

    def _parse_and_validate(self, raw_text: str) -> TOutput:
        """Parse a JSON LLM response and validate against the schema."""
        # Tolerate fenced code blocks like ```json ... ```
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            # Remove an optional language hint on the first line.
            first_newline = cleaned.find("\n")
            if first_newline != -1:
                cleaned = cleaned[first_newline + 1 :]
            cleaned = cleaned.strip().rstrip("`").strip()

        try:
            return self.output_schema.model_validate_json(cleaned)
        except ValidationError:
            logger.error(
                "LLM output failed schema validation for %s: %r",
                self.output_schema.__name__,
                raw_text[:500],
            )
            raise

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        input_rate, output_rate = _TIER_RATES_USD_PER_MILLION[self.tier]
        return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
