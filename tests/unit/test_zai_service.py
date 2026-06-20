import pytest
from unittest.mock import AsyncMock
from uuid import uuid4

from src.application.ports import LLMRequest, LLMResponse, UserMessageDTO
from src.application.zai.service import ZaiService
from src.domain.models import ChatRoles, LLMModelType
from src.infrastructure.gateways.zai_gateway import ZaiGateway


@pytest.mark.unit
class TestZaiService:
	@pytest.fixture
	def gateway(self) -> AsyncMock:
		gw = AsyncMock(spec=ZaiGateway)
		gw.generate.return_value = LLMResponse(
			text="glm reply",
			model=LLMModelType.glm_4_6,
			usage={"prompt_tokens": 5, "completion_tokens": 9},
			provider="zai",
		)
		return gw

	@pytest.fixture
	def service(self, gateway) -> ZaiService:
		return ZaiService(_gateway=gateway)

	@pytest.mark.asyncio
	async def test_returns_model_role_message(self, service, gateway):
		req = LLMRequest(
			message=UserMessageDTO(
				chat_id=uuid4(),
				message="hi",
				llm_model=LLMModelType.glm_4_6,
				role=ChatRoles.USER,
			)
		)
		result = await service.generate(req)

		assert result.role == ChatRoles.MODEL
		assert result.message == "glm reply"
		assert result.llm_model == LLMModelType.glm_4_6
		gateway.generate.assert_awaited_once()
