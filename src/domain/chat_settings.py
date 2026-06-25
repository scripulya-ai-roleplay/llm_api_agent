from enum import Enum
from typing import Optional

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
    contextLimitOverride: Optional[int] = Field(
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
