import asyncio
import logging
from typing import Annotated

import redis.asyncio
from dishka.integrations.faststream import FromDishka
from faststream import Context
from faststream.rabbit import RabbitRouter

from src.application.ports import IAgentService, LLMErrorResponse, LLMRequest, LLMResult, UserMessageDTO
from src.conf import settings
from src.domain.models import MODEL_PROVIDER_MAP
from src.infrastructure.exception_handler import ExceptionHandler
from src.infrastructure.heartbeat import Heartbeat, mark_done
from src.infrastructure.logging.trace import ensure_correlation_id, set_correlation_id

logger = logging.getLogger(__name__)

router = RabbitRouter()

# The backend sets this to a per-call request_id on publish; reusing it as the
# heartbeat key lets the watchdog correlate liveness to the right generation.
CorrelationId = Annotated[str, Context("message.correlation_id", default="")]


@router.subscriber(settings.LLM_REQUEST_QUEUE)
@router.publisher(settings.LLM_RESULT_QUEUE)
async def handle_llm_request(
	msg: LLMRequest,
	svc: FromDishka[IAgentService],
	handler: FromDishka[ExceptionHandler],
	redis_client: FromDishka[redis.asyncio.Redis],
	correlation_id: CorrelationId,
) -> LLMResult:
	set_correlation_id(correlation_id)
	ensure_correlation_id()

	logger.info("LLM request model=%s chat_id=%s", msg.message.llm_model, msg.message.chat_id)

	provider = MODEL_PROVIDER_MAP.get(msg.message.llm_model)
	provider_name = provider.value if provider is not None else None

	try:
		# The heartbeat lets the watchdog tell a slow-but-alive call from a dead agent;
		# the timeout catches a livelock the heartbeat can't (agent alive, provider
		# call never returns).
		async with Heartbeat(redis_client, correlation_id):
			async with asyncio.timeout(settings.LLM_GENERATION_TIMEOUT_SECONDS):
				reply: UserMessageDTO = await svc.handle(msg)
		logger.info("LLM ok chat_id=%s", reply.chat_id)
		return LLMResult(chat_id=msg.message.chat_id, message=reply)
	except TimeoutError:
		# Returned, not raised, so the failure still reaches the backend as a result.
		logger.warning(
			"LLM generation timed out after %ss chat_id=%s model=%s",
			settings.LLM_GENERATION_TIMEOUT_SECONDS,
			msg.message.chat_id,
			msg.message.llm_model,
		)
		return LLMResult(
			chat_id=msg.message.chat_id,
			error=LLMErrorResponse(
				error_code="generation_timeout",
				status=504,
				reason="Generation timed out",
				message=f"The model did not respond within {settings.LLM_GENERATION_TIMEOUT_SECONDS}s.",
				provider=provider_name,
			),
		)
	except Exception as exc:
		error = handler.handle(exc, provider=provider_name)
		return LLMResult(chat_id=msg.message.chat_id, error=error)
	finally:
		# Without this the watchdog would FAIL the generation once the heartbeat TTL lapses.
		await mark_done(redis_client, correlation_id)
