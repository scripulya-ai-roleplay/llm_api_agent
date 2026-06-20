import pytest
from unittest.mock import AsyncMock
from uuid import uuid4

from src.application.google.service import GoogleService
from src.application.ports import LLMRequest, LLMResponse, UserMessageDTO
from src.domain.models import ChatRoles, LLMModelType
from src.infrastructure.gateways.google_gateway import GoogleGateway


@pytest.mark.unit
class TestGoogleService:
	@pytest.fixture
	def gateway(self) -> AsyncMock:
		gw = AsyncMock(spec=GoogleGateway)
		gw.generate.return_value = LLMResponse(
			text="gemini reply",
			model=LLMModelType.gemini_flash_preview,
			usage={"prompt_token_count": 4},
			provider="google",
		)
		return gw

	@pytest.fixture
	def service(self, gateway) -> GoogleService:
		return GoogleService(_gateway=gateway)

	@pytest.mark.asyncio
	async def test_returns_model_role_message(self, service, gateway):
		req = LLMRequest(
			message=UserMessageDTO(
				chat_id=uuid4(),
				message="hello",
				llm_model=LLMModelType.gemini_flash_preview,
				role=ChatRoles.USER,
			)
		)
		result = await service.generate(req)

		assert result.role == ChatRoles.MODEL
		assert result.message == "gemini reply"
		assert result.chat_id == req.message.chat_id
		gateway.generate.assert_awaited_once()
		assert gateway.generate.call_args.kwargs["model"] == LLMModelType.gemini_flash_preview
