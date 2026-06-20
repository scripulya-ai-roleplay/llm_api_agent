import pytest
from unittest.mock import AsyncMock
from uuid import uuid4

from src.application.deepseek.service import DeepSeekService
from src.application.ports import LLMRequest, LLMResponse, UserMessageDTO
from src.domain.models import ChatRoles, LLMModelType
from src.infrastructure.gateways.deepseek_gateway import DeepSeekGateway


@pytest.mark.unit
class TestDeepSeekService:
	@pytest.fixture
	def gateway(self) -> AsyncMock:
		gw = AsyncMock(spec=DeepSeekGateway)
		gw.generate.return_value = LLMResponse(
			text="deepseek reply",
			model=LLMModelType.deepseek_chat,
			usage={"prompt_tokens": 6, "completion_tokens": 8},
			provider="deepseek",
		)
		return gw

	@pytest.fixture
	def service(self, gateway) -> DeepSeekService:
		return DeepSeekService(_gateway=gateway)

	@pytest.mark.asyncio
	async def test_returns_model_role_message(self, service, gateway):
		req = LLMRequest(
			message=UserMessageDTO(
				chat_id=uuid4(),
				message="hi",
				llm_model=LLMModelType.deepseek_chat,
				role=ChatRoles.USER,
			)
		)
		result = await service.generate(req)

		assert result.role == ChatRoles.MODEL
		assert result.message == "deepseek reply"
		assert result.llm_model == LLMModelType.deepseek_chat
		gateway.generate.assert_awaited_once()
