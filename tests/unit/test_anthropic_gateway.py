import logging
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import httpx
import pytest
from anthropic import APIStatusError

from src.application.ports import UserMessageDTO
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
)
from src.domain.models import ChatRoles, LLMModelType
from src.infrastructure.exceptions import AuthenticationException, ContentSafetyException
from src.infrastructure.gateways.anthropic_gateway import AnthropicGateway


def _status_error(status_code: int, body=None) -> APIStatusError:
	request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
	response = httpx.Response(status_code=status_code, request=request)
	return APIStatusError(message=f"err {status_code}", response=response, body=body or {})


async def _stream_from(items):
	for it in items:
		yield it


class _FakeMessageStream:
	"""Mimics the AsyncMessageStreamManager returned (synchronously) by messages.stream()."""

	def __init__(self, deltas: list[str] | None = None, final: SimpleNamespace = None, events: list | None = None):
		self._deltas = deltas or []
		self._final = final
		self._events = events
		self._it = None

	async def __aenter__(self):
		return self

	async def __aexit__(self, *exc):
		return False

	@property
	def text_stream(self):
		return _stream_from(self._deltas)

	def __aiter__(self):
		self._it = iter(self._events or [])
		return self

	async def __anext__(self):
		try:
			return next(self._it)
		except StopIteration:
			raise StopAsyncIteration

	async def get_final_message(self) -> SimpleNamespace:
		return self._final


def _reasoning_chat_settings(
	effort: ReasoningEffort = ReasoningEffort.MID, token_limit: TokenLimit = TokenLimit.MAX
) -> ChatSettings:
	return ChatSettings(
		aiControlBehavior=ControlBehavior.CONTROL,
		continueBehavior=ControlBehavior.CONTROL,
		perspective=Perspective.THIRD_PERSON,
		temperature=TemperatureSettings(preset=Preset.MID, value=0.7),
		responseLength=ResponseLength.MEDIUM,
		responseTokenLimit=token_limit,
		reasoning=Toggle.ON,
		reasoningEffort=effort,
		aiMediaPicker=Toggle.OFF,
		functions=FunctionsSettings(),
	)


def _final_message(text="hello world", stop_reason="end_turn") -> SimpleNamespace:
	return SimpleNamespace(
		content=[SimpleNamespace(type="text", text=text)],
		usage=SimpleNamespace(input_tokens=5, output_tokens=7),
		stop_reason=stop_reason,
	)


def _stream_client(final: SimpleNamespace) -> MagicMock:
	client = MagicMock()
	client.messages.stream = MagicMock(return_value=_FakeMessageStream(["hello", " ", "world"], final))
	return client


@pytest.mark.unit
class TestAnthropicGateway:
	@pytest.fixture
	def gateway(self) -> AnthropicGateway:
		return AnthropicGateway(logger=logging.getLogger(), _client=_stream_client(_final_message()))

	@pytest.mark.asyncio
	async def test_success_streams_tokens_and_extracts_text_and_usage(self, gateway):
		seen: list[str] = []

		async def collect(text: str) -> None:
			seen.append(text)

		result = await gateway.generate(
			model=LLMModelType.claude_sonnet,
			system_prompt="sys",
			user_message="hi",
			history=[],
			on_token=collect,
		)

		assert result.text == "hello world"
		assert result.provider == "anthropic"
		assert result.usage == {"input_tokens": 5, "output_tokens": 7}
		assert "".join(seen) == "hello world"

	@pytest.mark.asyncio
	async def test_history_mapped_to_anthropic_roles(self, gateway):
		history = [
			UserMessageDTO(chat_id=uuid4(), message="q1", llm_model=LLMModelType.claude_sonnet, role=ChatRoles.USER),
			UserMessageDTO(chat_id=uuid4(), message="a1", llm_model=LLMModelType.claude_sonnet, role=ChatRoles.MODEL),
			UserMessageDTO(
				chat_id=uuid4(), message="sys note", llm_model=LLMModelType.claude_sonnet, role=ChatRoles.SYSTEM
			),
		]
		await gateway.generate(
			model=LLMModelType.claude_sonnet,
			system_prompt="sys",
			user_message="q2",
			history=history,
		)
		kwargs = gateway._client.messages.stream.call_args.kwargs
		assert kwargs["model"] == "claude-sonnet-4-20250514"
		assert kwargs["system"] == "sys"
		assert kwargs["messages"] == [
			{"role": "user", "content": "q1"},
			{"role": "assistant", "content": "a1"},
			{"role": "user", "content": "q2"},
		]

	@pytest.mark.asyncio
	async def test_auth_error_maps_to_authentication_exception(self, gateway):
		gateway._client.messages.stream.side_effect = _status_error(401)
		with pytest.raises(AuthenticationException):
			await gateway.generate(
				model=LLMModelType.claude_sonnet,
				system_prompt="sys",
				user_message="hi",
				history=[],
			)

	@pytest.mark.asyncio
	async def test_content_filtered_maps_to_safety_exception(self):
		gateway = AnthropicGateway(
			logger=logging.getLogger(), _client=_stream_client(_final_message(stop_reason="content_filtered"))
		)
		with pytest.raises(ContentSafetyException):
			await gateway.generate(
				model=LLMModelType.claude_sonnet,
				system_prompt="sys",
				user_message="hi",
				history=[],
			)

	@pytest.mark.asyncio
	async def test_reasoning_splits_thinking_and_answer_and_forces_temperature(self):
		# With reasoning on, the gateway walks the raw event stream (not text_stream)
		# so thinking_delta blocks are not discarded; thinking goes to on_thinking,
		# answer text to on_token, and the reasoning is returned on the response.
		events = [
			SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="thinking_delta", thinking="delib")),
			SimpleNamespace(
				type="content_block_delta", delta=SimpleNamespace(type="thinking_delta", thinking="eration")
			),
			SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="hello ")),
			SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="world")),
			SimpleNamespace(type="message_stop"),
		]
		final = SimpleNamespace(
			content=[],
			usage=SimpleNamespace(input_tokens=5, output_tokens=7),
			stop_reason="end_turn",
		)
		client = MagicMock()
		client.messages.stream = MagicMock(return_value=_FakeMessageStream(final=final, events=events))
		gateway = AnthropicGateway(logger=logging.getLogger(), _client=client)

		tokens: list[str] = []
		thinking: list[str] = []

		async def on_token(text: str) -> None:
			tokens.append(text)

		async def on_thinking(text: str) -> None:
			thinking.append(text)

		result = await gateway.generate(
			model=LLMModelType.claude_sonnet,
			system_prompt="sys",
			user_message="hi",
			history=[],
			chat_settings=_reasoning_chat_settings(),
			on_token=on_token,
			on_thinking=on_thinking,
		)

		assert result.text == "hello world"
		assert result.reasoning == "deliberation"
		assert "".join(thinking) == "deliberation"
		assert "".join(tokens) == "hello world"

		# Extended thinking was requested with a budget, and temperature forced to 1.0.
		kwargs = client.messages.stream.call_args.kwargs
		assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 8192}
		assert kwargs["temperature"] == 1.0
