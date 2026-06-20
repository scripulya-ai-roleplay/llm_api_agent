import pytest
from unittest.mock import AsyncMock
from uuid import uuid4

from src.application.agent.service import AgentService
from src.application.ports import ILLMProviderService, LLMRequest, UserMessageDTO
from src.domain.models import ChatRoles, LLMModelType, LLMProvider
from src.infrastructure.exceptions import UnknownModelException


def _request(model: LLMModelType | None) -> LLMRequest:
	return LLMRequest(
		message=UserMessageDTO(
			chat_id=uuid4(),
			message="hi",
			llm_model=model,
			role=ChatRoles.USER,
		)
	)


@pytest.mark.unit
class TestAgentService:
	@pytest.fixture
	def services(self) -> dict[LLMProvider, AsyncMock]:
		return {p: AsyncMock(spec=ILLMProviderService) for p in LLMProvider}

	@pytest.fixture
	def agent(self, services) -> AgentService:
		return AgentService(provider_services=services)

	@pytest.mark.asyncio
	async def test_routes_each_provider(self, agent, services):
		cases = {
			LLMModelType.claude_sonnet: LLMProvider.ANTHROPIC,
			LLMModelType.gemini_flash_preview: LLMProvider.GOOGLE,
			LLMModelType.glm_4_6: LLMProvider.ZAI,
			LLMModelType.deepseek_chat: LLMProvider.DEEPSEEK,
			LLMModelType.testing_mock: LLMProvider.MOCK,
		}
		for model, provider in cases.items():
			req = _request(model)
			await agent.handle(req)
			services[provider].generate.assert_awaited_once_with(req)

	@pytest.mark.asyncio
	async def test_unknown_model_raises(self, agent):
		with pytest.raises(UnknownModelException):
			await agent.handle(_request(None))
