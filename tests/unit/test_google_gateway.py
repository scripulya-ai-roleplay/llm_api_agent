import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

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
from src.infrastructure.gateways.google_gateway import GoogleGateway


async def _chunk_stream(chunks):
	for chunk in chunks:
		yield chunk


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


def _gchunk(text: str | None) -> SimpleNamespace:
	return SimpleNamespace(
		text=text,
		prompt_feedback=SimpleNamespace(block_reason=None),
		usage_metadata=SimpleNamespace(prompt_token_count=3, candidates_token_count=4),
	)


def _gchunk_parts(parts: list[SimpleNamespace]) -> SimpleNamespace:
	# A chunk that carries explicit candidate parts (used when thinking_config splits
	# thought parts from answer parts); chunk.text is unused on the thinking path.
	return SimpleNamespace(
		candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))],
		prompt_feedback=SimpleNamespace(block_reason=None),
		usage_metadata=SimpleNamespace(prompt_token_count=3, candidates_token_count=4),
	)


def _stream_client(chunks: list[SimpleNamespace]) -> MagicMock:
	client = MagicMock()
	# generate_content_stream is `async def` in the SDK, so it is awaited to get the iterator.
	client.aio.models.generate_content_stream = AsyncMock(return_value=_chunk_stream(chunks))
	return client


@pytest.mark.unit
class TestGoogleGateway:
	@pytest.fixture
	def gateway(self) -> GoogleGateway:
		return GoogleGateway(
			logger=logging.getLogger(), _client=_stream_client([_gchunk("gemini "), _gchunk("says hi")])
		)

	@pytest.mark.asyncio
	async def test_success_streams_tokens_and_extracts_text_and_usage(self, gateway):
		seen: list[str] = []

		async def collect(text: str) -> None:
			seen.append(text)

		result = await gateway.generate(
			model=LLMModelType.gemini_flash_preview,
			system_prompt="sys",
			user_message="hi",
			history=[],
			on_token=collect,
		)

		assert result.text == "gemini says hi"
		assert result.provider == "google"
		assert result.usage == {"prompt_token_count": 3, "candidates_token_count": 4}
		assert seen == ["gemini ", "says hi"]

		kwargs = gateway._client.aio.models.generate_content_stream.call_args.kwargs
		assert kwargs["model"] == "gemini-3-flash-preview"
		# history empty -> contents is just the user message
		assert kwargs["contents"][-1] == "hi"

	@pytest.mark.asyncio
	async def test_history_mapped_to_contents(self, gateway):
		history = [
			UserMessageDTO(
				chat_id=uuid4(), message="u1", llm_model=LLMModelType.gemini_flash_preview, role=ChatRoles.USER
			),
			UserMessageDTO(
				chat_id=uuid4(), message="m1", llm_model=LLMModelType.gemini_flash_preview, role=ChatRoles.MODEL
			),
		]
		await gateway.generate(
			model=LLMModelType.gemini_flash_preview,
			system_prompt="sys",
			user_message="u2",
			history=history,
		)
		contents = gateway._client.aio.models.generate_content_stream.call_args.kwargs["contents"]
		# two Content objects from history + the final user string
		assert len(contents) == 3
		assert contents[-1] == "u2"

	@pytest.mark.asyncio
	async def test_auth_error_maps_to_authentication_exception(self, gateway):
		gateway._client.aio.models.generate_content_stream.side_effect = Exception("401 Permission denied")
		with pytest.raises(AuthenticationException):
			await gateway.generate(
				model=LLMModelType.gemini_flash_preview,
				system_prompt="sys",
				user_message="hi",
				history=[],
			)

	@pytest.mark.asyncio
	async def test_reasoning_splits_thought_parts_from_answer(self):
		# With include_thoughts, Gemini flags each thought part with part.thought=True; the
		# gateway routes those to on_thinking and the rest to on_token, and returns the
		# chain-of-thought as result.reasoning.
		chunks = [
			_gchunk_parts([SimpleNamespace(text="delib", thought=True)]),
			_gchunk_parts(
				[SimpleNamespace(text="eration", thought=True), SimpleNamespace(text="hello ", thought=False)]
			),
			_gchunk_parts([SimpleNamespace(text="world", thought=False)]),
		]
		gateway = GoogleGateway(logger=logging.getLogger(), _client=_stream_client(chunks))

		tokens: list[str] = []
		thinking: list[str] = []

		async def on_token(text: str) -> None:
			tokens.append(text)

		async def on_thinking(text: str) -> None:
			thinking.append(text)

		result = await gateway.generate(
			model=LLMModelType.gemini_flash_preview,
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

		config = gateway._client.aio.models.generate_content_stream.call_args.kwargs["config"]
		assert config.thinking_config.include_thoughts is True
		assert config.thinking_config.thinking_budget == 8192  # MID effort under MAX cap

	@pytest.mark.asyncio
	async def test_reasoning_off_omits_thinking_config(self, gateway):
		await gateway.generate(
			model=LLMModelType.gemini_flash_preview,
			system_prompt="sys",
			user_message="hi",
			history=[],
		)
		config = gateway._client.aio.models.generate_content_stream.call_args.kwargs["config"]
		assert config.thinking_config is None
