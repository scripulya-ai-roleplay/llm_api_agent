import logging
from dataclasses import dataclass

from src.application.ports import ILLMProviderService, LLMRequest, LLMResponse, UserMessageDTO
from src.domain.models import ChatRoles
from src.infrastructure.gateways.mock_gateway import MockGateway

logger = logging.getLogger(__name__)


@dataclass
class MockService(ILLMProviderService):
	_gateway: MockGateway

	async def generate(self, request: LLMRequest) -> UserMessageDTO:
		resp: LLMResponse = await self._gateway.generate(
			model=request.message.llm_model,
			system_prompt="",
			user_message=request.message.message,
			history=request.history,
		)
		return UserMessageDTO(
			chat_id=request.message.chat_id,
			message=resp.text,
			llm_model=request.message.llm_model,
			role=ChatRoles.MODEL,
		)
