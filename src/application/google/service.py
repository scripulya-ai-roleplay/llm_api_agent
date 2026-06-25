import logging
from dataclasses import dataclass

from src.application.ports import ILLMProviderService, LLMRequest, LLMResponse, UserMessageDTO
from src.conf import settings
from src.domain.models import ChatRoles
from src.infrastructure.gateways.google_gateway import GoogleGateway

logger = logging.getLogger(__name__)


@dataclass
class GoogleService(ILLMProviderService):
	_gateway: GoogleGateway

	async def generate(self, request: LLMRequest) -> UserMessageDTO:
		resp: LLMResponse = await self._gateway.generate(
			model=request.message.llm_model,
			system_prompt=settings.SYSTEM_PROMPT,
			user_message=request.message.message,
			history=request.history,
			chat_settings=request.chat_settings,
		)
		logger.info("google ok model=%s usage=%s chat_id=%s", resp.model, resp.usage, request.message.chat_id)
		return UserMessageDTO(
			chat_id=request.message.chat_id,
			message=resp.text,
			llm_model=request.message.llm_model,
			role=ChatRoles.MODEL,
		)
