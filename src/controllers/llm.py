import logging
from typing import Annotated

from dishka.integrations.faststream import FromDishka
from faststream import Context
from faststream.rabbit import RabbitRouter

from src.application.ports import IAgentService, LLMRequest, LLMResult, UserMessageDTO
from src.conf import settings
from src.domain.models import MODEL_PROVIDER_MAP
from src.infrastructure.exception_handler import ExceptionHandler
from src.infrastructure.logging.trace import ensure_correlation_id, set_correlation_id

logger = logging.getLogger(__name__)

router = RabbitRouter()

# Correlation id propagated by RabbitMQ (set by the backend on publish).
CorrelationId = Annotated[str, Context("message.correlation_id", default="")]


@router.subscriber(settings.LLM_REQUEST_QUEUE)
@router.publisher(settings.LLM_RESULT_QUEUE)
async def handle_llm_request(
	msg: LLMRequest,
	svc: FromDishka[IAgentService],
	handler: FromDishka[ExceptionHandler],
	correlation_id: CorrelationId,
) -> LLMResult:
	"""Consume an LLMRequest, dispatch to the provider service, and publish
	an LLMResult (carrying the model's reply UserMessageDTO or an error) to
	the result queue.
	"""
	set_correlation_id(correlation_id)
	ensure_correlation_id()  # backfill a generated id if the backend omitted one

	logger.info("LLM request model=%s chat_id=%s", msg.message.llm_model, msg.message.chat_id)

	provider = MODEL_PROVIDER_MAP.get(msg.message.llm_model)
	provider_name = provider.value if provider is not None else None

	try:
		reply: UserMessageDTO = await svc.handle(msg)  # role=MODEL, same chat_id/llm_model
		logger.info("LLM ok chat_id=%s", reply.chat_id)
		return LLMResult(chat_id=msg.message.chat_id, message=reply)
	except Exception as exc:
		# All failures (domain AgentException subclasses + any unexpected error)
		# are funneled through the central handler, which logs and builds the
		# structured LLMErrorResponse payload for the backend.
		error = handler.handle(exc, provider=provider_name)
		return LLMResult(chat_id=msg.message.chat_id, error=error)
