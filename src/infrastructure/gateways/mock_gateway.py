import asyncio
from dataclasses import dataclass
from logging import Logger
from typing import ClassVar

from src.application.ports import ILLMProviderGateway, LLMResponse, UserMessageDTO
from src.domain.chat_settings import ChatSettings
from src.domain.models import LLMModelType, LLMProvider
from src.infrastructure.gateways._streaming import emit_token


def _word_chunks(text: str) -> list[str]:
	pieces = text.split(" ")
	return [p + (" " if i < len(pieces) - 1 else "") for i, p in enumerate(pieces)]


@dataclass
class MockGateway(ILLMProviderGateway):
	"""Offline stand-in used when `llm_model == testing_mock`. No API key required."""

	provider: ClassVar[LLMProvider] = LLMProvider.MOCK

	logger: Logger

	async def generate(
		self,
		model: LLMModelType,
		system_prompt: str,  # noqa: ARG002 - unused by the mock
		user_message: str,
		history: list[UserMessageDTO],  # noqa: ARG002 - unused by the mock
		chat_settings: ChatSettings | None = None,  # noqa: ARG002 - unused by the mock
		on_token=None,
		on_thinking=None,  # noqa: ARG002 - the mock never thinks
	) -> LLMResponse:
		self.logger.info("Mock gateway received: %s", user_message)
		text = f"Mock response for: {user_message}"
		if on_token is not None:
			for chunk in _word_chunks(text):
				await emit_token(on_token, chunk)
				await asyncio.sleep(0.01)
		return LLMResponse(
			text=text,
			model=LLMModelType.testing_mock,
			usage={"tokens": 10},
			provider=LLMProvider.MOCK.value,
		)
