import pytest

from src.conf import settings
from src.domain.chat_settings import (
	ChatSettings,
	ControlBehavior,
	FunctionsSettings,
	Perspective,
	Preset,
	ReasoningEffort,
	ResponseLength,
	TemperatureSettings,
	TokenLimit,
	Toggle,
	resolve_max_tokens,
	resolve_temperature,
)


def _settings(token_limit: TokenLimit, temperature: float) -> ChatSettings:
	return ChatSettings(
		aiControlBehavior=ControlBehavior.CONTROL,
		continueBehavior=ControlBehavior.CONTROL,
		perspective=Perspective.THIRD_PERSON,
		temperature=TemperatureSettings(preset=Preset.MID, value=temperature),
		responseLength=ResponseLength.MEDIUM,
		responseTokenLimit=token_limit,
		reasoning=Toggle.OFF,
		reasoningEffort=ReasoningEffort.MID,
		aiMediaPicker=Toggle.OFF,
		functions=FunctionsSettings(),
	)


@pytest.mark.unit
class TestResolve:
	def test_temperature_falls_back_to_default_when_none(self):
		assert resolve_temperature(None) == settings.LLM_TEMPERATURE

	def test_temperature_uses_chat_value(self):
		assert resolve_temperature(_settings(TokenLimit.HIGH, 0.9)) == 0.9

	def test_max_tokens_falls_back_to_default_when_none(self):
		assert resolve_max_tokens(None) == settings.LLM_MAX_TOKENS

	@pytest.mark.parametrize(
		("limit", "expected"),
		[
			(TokenLimit.CAPPED, 2048),
			(TokenLimit.HIGH, 8192),
			(TokenLimit.MAX, 16384),
		],
	)
	def test_max_tokens_resolved_from_preset(self, limit, expected):
		assert resolve_max_tokens(_settings(limit, 0.5)) == expected
