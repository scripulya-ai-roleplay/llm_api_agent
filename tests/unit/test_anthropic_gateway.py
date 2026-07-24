import logging
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import httpx
import pytest
from anthropic import APIStatusError

from src.application.ports import UserMessageDTO
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

	def __init__(self, deltas: list[str], final: SimpleNamespace):
		self._deltas = deltas
		self._final = final

	async def __aenter__(self):
		return self

	async def __aexit__(self, *exc):
		return False

	@property
	def text_stream(self):
		return _stream_from(self._deltas)

	async def get_final_message(self) -> SimpleNamespace:
		return self._final


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
