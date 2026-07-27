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
	Toggle,
	TokenLimit,
	reasoning_enabled,
	resolve_max_tokens,
	resolve_temperature,
	resolve_thinking_budget,
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


def _reasoning_settings(effort: ReasoningEffort, token_limit: TokenLimit) -> ChatSettings:
	s = _settings(token_limit, 0.7)
	return s.model_copy(update={"reasoning": Toggle.ON, "reasoningEffort": effort})


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

	def test_reasoning_enabled_is_false_without_settings(self):
		assert reasoning_enabled(None) is False

	def test_reasoning_enabled_reflects_toggle(self):
		assert reasoning_enabled(_settings(TokenLimit.HIGH, 0.5)) is False
		assert reasoning_enabled(_reasoning_settings(ReasoningEffort.MID, TokenLimit.HIGH)) is True

	def test_thinking_budget_none_when_reasoning_off(self):
		assert resolve_thinking_budget(_settings(TokenLimit.HIGH, 0.5)) is None

	def test_thinking_budget_uses_effort_when_under_cap(self):
		# MID = 8192, comfortably under MAX (16384).
		assert resolve_thinking_budget(_reasoning_settings(ReasoningEffort.MID, TokenLimit.MAX)) == 8192

	def test_thinking_budget_clamps_to_leave_room_for_answer(self):
		# HIGH = 16000 but cap is HIGH (8192); clamp to cap - 1024.
		assert resolve_thinking_budget(_reasoning_settings(ReasoningEffort.HIGH, TokenLimit.HIGH)) == 7168
