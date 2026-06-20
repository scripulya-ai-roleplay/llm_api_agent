from dataclasses import dataclass
from logging import Logger
from typing import ClassVar

from src.application.ports import ILLMProviderGateway, LLMResponse, UserMessageDTO
from src.domain.models import LLMModelType, LLMProvider


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
	) -> LLMResponse:
		self.logger.info("Mock gateway received: %s", user_message)
		return LLMResponse(
			text=f"Mock response for: {user_message}",
			model=LLMModelType.testing_mock,
			usage={"tokens": 10},
			provider=LLMProvider.MOCK.value,
		)
