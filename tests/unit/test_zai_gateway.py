import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
import pytest
from openai import APIStatusError

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
from src.infrastructure.exceptions import AuthenticationException
from src.infrastructure.gateways.zai_gateway import ZaiGateway


def _status_error(status_code: int) -> APIStatusError:
	request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
	response = httpx.Response(status_code=status_code, request=request)
	return APIStatusError(message=f"err {status_code}", response=response, body={})


async def _chunk_stream(chunks):
	for chunk in chunks:
		yield chunk


def _delta(content: str | None, finish_reason: str | None = None) -> SimpleNamespace:
	return SimpleNamespace(
		choices=[SimpleNamespace(delta=SimpleNamespace(content=content), finish_reason=finish_reason)]
	)


def _rdelta(
	reasoning: str | None = None, content: str | None = None, finish_reason: str | None = None
) -> SimpleNamespace:
	# delta carries reasoning_content only when the model emits a thinking chunk; plain
	# answer chunks omit it (so getattr falls back to None, as with the real SDK).
	delta = SimpleNamespace(content=content)
	if reasoning is not None:
		delta.reasoning_content = reasoning
	return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)])


def _reasoning_chat_settings() -> ChatSettings:
	return ChatSettings(
		aiControlBehavior=ControlBehavior.CONTROL,
		continueBehavior=ControlBehavior.CONTROL,
		perspective=Perspective.THIRD_PERSON,
		temperature=TemperatureSettings(preset=Preset.MID, value=0.7),
		responseLength=ResponseLength.MEDIUM,
		responseTokenLimit=TokenLimit.MAX,
		reasoning=Toggle.ON,
		reasoningEffort=ReasoningEffort.MID,
		aiMediaPicker=Toggle.OFF,
		functions=FunctionsSettings(),
	)


def _chunks(deltas: list[str], finish="stop") -> list[SimpleNamespace]:
	out = [_delta(d) for d in deltas]
	out.append(_delta(None, finish_reason=finish))
	return out


def _stream_client(chunks: list[SimpleNamespace]) -> MagicMock:
	client = MagicMock()
	client.chat.completions.create = AsyncMock(return_value=_chunk_stream(chunks))
	return client


@pytest.mark.unit
class TestZaiGateway:
	@pytest.fixture
	def gateway(self) -> ZaiGateway:
		return ZaiGateway(logger=logging.getLogger(), _client=_stream_client(_chunks(["zai", " reply"])))

	@pytest.mark.asyncio
	async def test_success_streams_tokens_and_accumulates_text(self, gateway):
		seen: list[str] = []

		async def collect(text: str) -> None:
			seen.append(text)

		result = await gateway.generate(
			model=LLMModelType.glm_4_6,
			system_prompt="sys",
			user_message="hi",
			history=[],
			on_token=collect,
		)

		assert result.text == "zai reply"
		assert result.provider == "zai"
		assert result.usage is None  # stream_options omitted; usage unavailable on the streaming path
		assert seen == ["zai", " reply"]

		kwargs = gateway._client.chat.completions.create.call_args.kwargs
		assert kwargs["model"] == "glm-4.6"
		assert kwargs["stream"] is True
		# first message is the system prompt, last is the user message
		assert kwargs["messages"][0] == {"role": "system", "content": "sys"}
		assert kwargs["messages"][-1] == {"role": "user", "content": "hi"}

	@pytest.mark.asyncio
	async def test_history_roles_mapped(self, gateway):
		history = [
			UserMessageDTO(chat_id=uuid4(), message="u1", llm_model=LLMModelType.glm_4_6, role=ChatRoles.USER),
			UserMessageDTO(chat_id=uuid4(), message="m1", llm_model=LLMModelType.glm_4_6, role=ChatRoles.MODEL),
		]
		await gateway.generate(
			model=LLMModelType.glm_4_6,
			system_prompt="sys",
			user_message="u2",
			history=history,
		)
		messages = gateway._client.chat.completions.create.call_args.kwargs["messages"]
		roles = [m["role"] for m in messages]
		assert roles == ["system", "user", "assistant", "user"]

	@pytest.mark.asyncio
	async def test_auth_error_maps_to_authentication_exception(self, gateway):
		gateway._client.chat.completions.create.side_effect = _status_error(401)
		with pytest.raises(AuthenticationException):
			await gateway.generate(
				model=LLMModelType.glm_4_6,
				system_prompt="sys",
				user_message="hi",
				history=[],
			)

	@pytest.mark.asyncio
	async def test_reasoning_routes_reasoning_content_to_separate_sink(self):
		# GLM's chain-of-thought arrives as delta.reasoning_content (DeepSeek-compatible);
		# it must stream via on_thinking and be returned as result.reasoning, while the
		# answer still streams via on_token. The reasoning toggle also opts into the
		# `thinking` param via extra_body.
		chunks = [
			_rdelta(reasoning="delib"),
			_rdelta(reasoning="eration"),
			_rdelta(content="hello "),
			_rdelta(content="world", finish_reason="stop"),
		]
		gateway = ZaiGateway(logger=logging.getLogger(), _client=_stream_client(chunks))

		tokens: list[str] = []
		thinking: list[str] = []

		async def on_token(text: str) -> None:
			tokens.append(text)

		async def on_thinking(text: str) -> None:
			thinking.append(text)

		result = await gateway.generate(
			model=LLMModelType.glm_4_6,
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

		kwargs = gateway._client.chat.completions.create.call_args.kwargs
		assert kwargs["extra_body"]["thinking"] == {"type": "enabled"}
		assert kwargs["extra_body"]["reasoning_effort"] == "medium"

	@pytest.mark.asyncio
	async def test_reasoning_off_omits_thinking_param(self, gateway):
		await gateway.generate(
			model=LLMModelType.glm_4_6,
			system_prompt="sys",
			user_message="hi",
			history=[],
		)
		kwargs = gateway._client.chat.completions.create.call_args.kwargs
		assert "extra_body" not in kwargs
