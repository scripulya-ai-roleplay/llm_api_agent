import pytest
from unittest.mock import AsyncMock
from uuid import uuid4

from src.application.anthropic.service import AnthropicService
from src.application.ports import LLMRequest, LLMResponse, UserMessageDTO
from src.conf import settings
from src.domain.models import ChatRoles, LLMModelType
from src.infrastructure.gateways.anthropic_gateway import AnthropicGateway


@pytest.mark.unit
class TestAnthropicService:
	@pytest.fixture
	def gateway(self) -> AsyncMock:
		gw = AsyncMock(spec=AnthropicGateway)
		gw.generate.return_value = LLMResponse(
			text="hello there",
			model=LLMModelType.claude_sonnet,
			usage={"input_tokens": 3, "output_tokens": 2},
			provider="anthropic",
		)
		return gw

	@pytest.fixture
	def service(self, gateway) -> AnthropicService:
		return AnthropicService(_gateway=gateway)

	@pytest.fixture
	def llm_request(self) -> LLMRequest:
		return LLMRequest(
			message=UserMessageDTO(
				chat_id=uuid4(),
				message="hi",
				llm_model=LLMModelType.claude_sonnet,
				role=ChatRoles.USER,
			),
			history=[
				UserMessageDTO(
					chat_id=uuid4(), message="earlier", llm_model=LLMModelType.claude_sonnet, role=ChatRoles.MODEL
				),
			],
		)

	@pytest.mark.asyncio
	async def test_returns_model_role_message(self, service, gateway, llm_request):
		result = await service.generate(llm_request)

		assert result.role == ChatRoles.MODEL
		assert result.message == "hello there"
		assert result.chat_id == llm_request.message.chat_id
		assert result.llm_model == LLMModelType.claude_sonnet

		gateway.generate.assert_awaited_once()
		kwargs = gateway.generate.call_args.kwargs
		assert kwargs["model"] == LLMModelType.claude_sonnet
		assert kwargs["system_prompt"] == settings.SYSTEM_PROMPT
		assert kwargs["user_message"] == "hi"
		assert kwargs["history"] == llm_request.history
