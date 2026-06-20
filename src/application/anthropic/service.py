import logging
from dataclasses import dataclass

from src.application.ports import ILLMProviderService, LLMRequest, LLMResponse, UserMessageDTO
from src.conf import settings
from src.domain.models import ChatRoles
from src.infrastructure.gateways.anthropic_gateway import AnthropicGateway

logger = logging.getLogger(__name__)


@dataclass
class AnthropicService(ILLMProviderService):
	_gateway: AnthropicGateway

	async def generate(self, request: LLMRequest) -> UserMessageDTO:
		resp: LLMResponse = await self._gateway.generate(
			model=request.message.llm_model,
			system_prompt=settings.SYSTEM_PROMPT,
			user_message=request.message.message,
			history=request.history,
		)
		logger.info(
			"anthropic ok model=%s usage=%s chat_id=%s",
			resp.model,
			resp.usage,
			request.message.chat_id,
		)
		return UserMessageDTO(
			chat_id=request.message.chat_id,
			message=resp.text,
			llm_model=request.message.llm_model,
			role=ChatRoles.MODEL,
		)
