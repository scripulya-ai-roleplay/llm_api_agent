import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domain.models import LLMModelType
from src.infrastructure.gateways.deepseek_gateway import DeepSeekGateway


async def _chunk_stream(chunks):
	for chunk in chunks:
		yield chunk


def _delta(content: str | None, finish_reason: str | None = None) -> SimpleNamespace:
	return SimpleNamespace(
		choices=[SimpleNamespace(delta=SimpleNamespace(content=content), finish_reason=finish_reason)],
		usage=None,
	)


def _usage_chunk(prompt=10, completion=20, total=30) -> SimpleNamespace:
	# OpenAI streaming delivers usage on a final chunk with empty choices.
	return SimpleNamespace(
		usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total),
		choices=[],
	)


def _chunks(deltas: list[str], finish="stop") -> list[SimpleNamespace]:
	out = [_delta(d) for d in deltas]
	out.append(_delta(None, finish_reason=finish))
	out.append(_usage_chunk())
	return out


def _stream_client(chunks: list[SimpleNamespace]) -> MagicMock:
	client = MagicMock()
	client.chat.completions.create = AsyncMock(return_value=_chunk_stream(chunks))
	return client


@pytest.mark.unit
class TestDeepSeekGateway:
	@pytest.fixture
	def gateway(self) -> DeepSeekGateway:
		return DeepSeekGateway(logger=logging.getLogger(), _client=_stream_client(_chunks(["deepseek", " reply"])))

	@pytest.mark.asyncio
	async def test_success_streams_tokens_and_accumulates_text(self, gateway):
		seen: list[str] = []

		async def collect(text: str) -> None:
			seen.append(text)

		result = await gateway.generate(
			model=LLMModelType.deepseek_chat,
			system_prompt="sys",
			user_message="hi",
			history=[],
			on_token=collect,
		)

		assert result.text == "deepseek reply"
		assert result.provider == "deepseek"
		assert result.usage == {
			"prompt_tokens": 10,
			"completion_tokens": 20,
			"total_tokens": 30,
		}
		assert seen == ["deepseek", " reply"]

		kwargs = gateway._client.chat.completions.create.call_args.kwargs
		assert kwargs["model"] == "deepseek-chat"
		assert kwargs["stream"] is True
		assert kwargs["stream_options"] == {"include_usage": True}
