import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domain.models import LLMModelType
from src.infrastructure.gateways.deepseek_gateway import DeepSeekGateway


def _fake_response(content="deepseek reply") -> SimpleNamespace:
	return SimpleNamespace(
		choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
		usage=SimpleNamespace(prompt_tokens=6, completion_tokens=8),
	)


@pytest.mark.unit
class TestDeepSeekGateway:
	@pytest.fixture
	def gateway(self) -> DeepSeekGateway:
		client = MagicMock()
		client.chat.completions.create = AsyncMock(return_value=_fake_response())
		return DeepSeekGateway(logger=logging.getLogger(), _client=client)

	@pytest.mark.asyncio
	async def test_success_extracts_text_and_usage(self, gateway):
		result = await gateway.generate(
			model=LLMModelType.deepseek_chat,
			system_prompt="sys",
			user_message="hi",
			history=[],
		)
		assert result.text == "deepseek reply"
		assert result.provider == "deepseek"
		assert result.usage == {"prompt_tokens": 6, "completion_tokens": 8}

		kwargs = gateway._client.chat.completions.create.call_args.kwargs
		assert kwargs["model"] == "deepseek-chat"
