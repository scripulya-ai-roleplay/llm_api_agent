from enum import Enum

from pydantic import BaseModel, Field

from src.conf import settings


class ControlBehavior(str, Enum):
	CONTROL = "Control"
	DONT_CONTROL = "Don't Control"


class Perspective(str, Enum):
	FIRST_PERSON = "1st Person"
	SECOND_PERSON = "2nd Person"
	THIRD_PERSON = "3rd Person"


class Preset(str, Enum):
	LOW = "Low"
	MID = "Mid"
	HIGH = "High"
	MAX = "Max"


class ResponseLength(str, Enum):
	SHORT = "Short"
	MEDIUM = "Medium"
	LONG = "Long"


class TokenLimit(str, Enum):
	CAPPED = "Capped"
	HIGH = "High"
	MAX = "Max"


class Toggle(str, Enum):
	ON = "On"
	OFF = "Off"


class ReasoningEffort(str, Enum):
	MIN = "Min"
	LOW = "Low"
	MID = "Mid"
	HIGH = "High"


class TemperatureSettings(BaseModel):
	preset: Preset
	value: float = Field(..., ge=0.0, description="Controls AI creativity")


class FunctionsSettings(BaseModel):
	characterNameGenerator: bool = Field(default=True, description="Generates unique character names using AI")


class ChatSettings(BaseModel):
	"""Per-chat LLM generation settings received from the backend.

	Mirror of the backend's ChatSettings (src/application/chats/settings.py).
	Kept verbatim because it crosses the RabbitMQ boundary as JSON and must
	deserialize identically on both sides. Only the universal knobs
	(temperature, responseTokenLimit) are translated into provider calls today;
	the remaining fields travel through every layer so per-provider enablement
	(reasoning, perspective, ...) can be added later without touching the contract.
	"""

	aiControlBehavior: ControlBehavior
	continueBehavior: ControlBehavior
	perspective: Perspective
	temperature: TemperatureSettings
	responseLength: ResponseLength
	responseTokenLimit: TokenLimit = Field(description="Max token limit. 2k tokens noted in UI.")
	reasoning: Toggle
	reasoningEffort: ReasoningEffort
	aiMediaPicker: Toggle
	contextLimitOverride: int | None = Field(
		default=None,
		ge=1,
		le=1048576,
		description="Set context limit to save cost. Max 1,048,576.",
	)
	functions: FunctionsSettings


# --- Resolution of the universal knobs into provider-call arguments ---------

# responseTokenLimit presets -> concrete output-token caps.
# CAPPED ~= 2k tokens (per the UI note); HIGH/MAX scale up from there.
_TOKEN_LIMIT: dict[TokenLimit, int] = {
	TokenLimit.CAPPED: 2048,
	TokenLimit.HIGH: 8192,
	TokenLimit.MAX: 16384,
}


def resolve_temperature(chat_settings: ChatSettings | None) -> float:
	"""Concrete temperature for a provider call. Falls back to the agent default
	when the chat carries no settings (backward compatible)."""
	if chat_settings and chat_settings.temperature is not None:
		return chat_settings.temperature.value
	return settings.LLM_TEMPERATURE


def resolve_max_tokens(chat_settings: ChatSettings | None) -> int:
	"""Concrete max output-token cap for a provider call, resolved from the
	chat's responseTokenLimit preset. Falls back to the agent default otherwise."""
	if chat_settings and chat_settings.responseTokenLimit is not None:
		return _TOKEN_LIMIT.get(chat_settings.responseTokenLimit, settings.LLM_MAX_TOKENS)
	return settings.LLM_MAX_TOKENS


def reasoning_enabled(chat_settings: ChatSettings | None) -> bool:
	"""Whether the chat has opted into reasoning (extended thinking).

	A reasoner model like deepseek-reasoner thinks unconditionally; this flag is
	the explicit opt-in for providers where thinking is a billable, latency-adding
	mode (Anthropic)."""
	return bool(chat_settings and chat_settings.reasoning == Toggle.ON)


_THINKING_BUDGET_SHARE: dict[ReasoningEffort, float] = {
	ReasoningEffort.MIN: 0.0,
	ReasoningEffort.LOW: 0.25,
	ReasoningEffort.MID: 0.5,
	ReasoningEffort.HIGH: 1.0,
}

_ANTHROPIC_MIN_THINKING_BUDGET = 1024


def resolve_thinking_budget(chat_settings: ChatSettings | None) -> int | None:
	"""Concrete thinking budget (tokens) for a provider call when reasoning is on.

	The budget is a share of the chat's output-token cap so the effort setting
	stays meaningful at every responseTokenLimit: MIN adds only the provider
	minimum, HIGH can use up to the whole cap. Anthropic additionally requires
	budget_tokens in [1024, max_tokens); the upper bound keeps at least the
	provider minimum for the answer itself. Returns None when reasoning is off."""
	if not reasoning_enabled(chat_settings):
		return None
	share = _THINKING_BUDGET_SHARE.get(chat_settings.reasoningEffort, 0.5)
	cap = resolve_max_tokens(chat_settings)
	budget = int(cap * share)
	return min(max(budget, _ANTHROPIC_MIN_THINKING_BUDGET), cap - _ANTHROPIC_MIN_THINKING_BUDGET)


def resolve_reasoning_max_tokens(chat_settings: ChatSettings | None) -> int:
	"""Output cap for a reasoning call: the chat's cap raised just enough that the
	effort-derived thinking budget fits under it with room for the answer
	(Anthropic requires budget_tokens < max_tokens)."""
	cap = resolve_max_tokens(chat_settings)
	budget = resolve_thinking_budget(chat_settings)
	if budget is None:
		return cap
	return max(cap, budget + _ANTHROPIC_MIN_THINKING_BUDGET)


_ZAI_EFFORT: dict[ReasoningEffort, str] = {
	ReasoningEffort.MIN: "min",
	ReasoningEffort.LOW: "low",
	ReasoningEffort.MID: "medium",
	ReasoningEffort.HIGH: "high",
}


def resolve_reasoning_effort(chat_settings: ChatSettings | None) -> str | None:
	"""Provider-level effort string for OpenAI-compatible deep-thinking params
	(Z.ai GLM 5.2+: `reasoning_effort`, effective only when thinking is enabled).

	Returns None when reasoning is off or no effort is mapped; the caller then
	omits the parameter and the provider default applies."""
	if not reasoning_enabled(chat_settings):
		return None
	return _ZAI_EFFORT.get(chat_settings.reasoningEffort) if chat_settings else None
