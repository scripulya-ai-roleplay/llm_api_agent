import logging
from dataclasses import dataclass

from src.application.ports import IAgentService, ILLMProviderService, LLMRequest, UserMessageDTO
from src.domain.models import MODEL_PROVIDER_MAP, LLMProvider
from src.infrastructure.exceptions import UnknownModelException

logger = logging.getLogger(__name__)


@dataclass
class AgentService(IAgentService):
	"""Routes `request.message.llm_model` to the correct provider service."""

	provider_services: dict[LLMProvider, ILLMProviderService]

	async def handle(self, request: LLMRequest) -> UserMessageDTO:
		model = request.message.llm_model
		provider = MODEL_PROVIDER_MAP.get(model)
		if provider is None:
			raise UnknownModelException(
				message=f"Model '{model}' is not mapped to any provider",
				details={"model": str(model)},
			)
		service = self.provider_services.get(provider)
		if service is None:
			raise UnknownModelException(
				message=f"Provider '{provider}' has no registered service",
				details={"provider": str(provider)},
			)
		logger.info("Dispatching model=%s -> provider=%s", model, provider)
		return await service.generate(request)
