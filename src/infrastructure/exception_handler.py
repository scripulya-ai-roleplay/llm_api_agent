import logging
from dataclasses import dataclass
from logging import Logger
from typing import Any

from src.application.ports import LLMErrorResponse
from src.infrastructure.exceptions import (
	AgentException,
	AuthenticationException,
	LLMGatewayException,
	RateLimitException,
)

logger = logging.getLogger(__name__)

# Message tokens used to infer an HTTP status when the provider SDK does not
# expose one (e.g. google-genai, or OpenAI/Anthropic transport errors).
_AUTH_HINTS: tuple[str, ...] = (
	"401", "403", "permission", "unauthenticated", "unauthorized",
	"api key", "api_key", "credential", "credentials", "no api key",
)
_RATE_HINTS: tuple[str, ...] = ("429", "quota", "rate limit", "resource_exhausted", "rate_limit")


def _infer_status(exc: Exception) -> int | None:
	"""Best-effort guess at an HTTP status from an exception's message."""
	text = str(exc).lower()
	if any(hint in text for hint in _AUTH_HINTS):
		return 401
	if any(hint in text for hint in _RATE_HINTS):
		return 429
	return None


def _provider_error_details(exc: Exception, status: int | None, body: Any) -> dict[str, Any]:
	"""Uniform `details` payload for a classified provider error."""
	details: dict[str, Any] = {"status": status, "error_type": type(exc).__name__}
	if body:
		details["body"] = str(body)
	return details


@dataclass
class ExceptionHandler:
	"""Single choke point for exception handling in the project.

	Two responsibilities:
	- `classify_provider_error`: translate a raw provider SDK error into the right
	  domain `AgentException` (centralizing the status -> exception mapping shared
	  by every gateway).
	- `handle`: serialize any exception into the `LLMErrorResponse` payload
	  published to RabbitMQ.
	"""

	logger: Logger

	@staticmethod
	def classify_provider_error(
		exc: Exception,
		*,
		provider: str,
		status: int | None = None,
		body: Any = None,
	) -> AgentException:
		"""Map a raw provider SDK error to the right domain `AgentException`.

		Callers that already have a concrete HTTP status (OpenAI/Anthropic
		`status_code`) pass it in; when it is missing (e.g. google-genai, or a
		transport-level error) the message is inspected for auth/rate-limit hints
		before falling back to a generic gateway error.
		"""
		if status is None:
			status = _infer_status(exc)
		logger.error("%s provider error: %s status=%s body=%s", provider, exc, status, body)
		details = _provider_error_details(exc, status, body)

		if status in (401, 403):
			return AuthenticationException(message=f"{provider} auth failed ({status}): {exc}", details=details)
		if status == 429:
			return RateLimitException(message=f"{provider} rate limit / quota exceeded: {exc}", details=details)
		if status is None:
			return LLMGatewayException(message=f"{provider} gateway error: {exc}", details=details)
		return LLMGatewayException(message=f"{provider} API error ({status}): {exc}", details=details)

	def handle(self, exc: Exception, *, provider: str | None = None) -> LLMErrorResponse:
		if isinstance(exc, AgentException):
			self.logger.warning(
				"Handled error code=%s status=%s provider=%s: %s",
				exc.error_code,
				exc.status,
				provider,
				exc.message,
			)
			return LLMErrorResponse(
				error_code=exc.error_code,
				status=exc.status,
				reason=exc.reason,
				message=exc.message,
				provider=provider,
				details=exc.details,
			)

		# Unexpected (non-domain) exception: classify it so the real cause and a
		# meaningful error_code/status surface to the caller instead of a generic
		# "Internal agent error". The full traceback is logged above for debugging.
		self.logger.exception("Unhandled error provider=%s: %s", provider, exc)
		classified = self.classify_provider_error(exc, provider=provider or "unknown")
		return LLMErrorResponse(
			error_code=classified.error_code,
			status=classified.status,
			reason=classified.reason,
			message=classified.message,
			provider=provider,
			details=classified.details,
		)
