from typing import Any


class AgentException(Exception):
	"""Base for all service-level exceptions.

	Each subclass declares a stable machine-readable `error_code`, an HTTP-style
	`status`, and a short human `reason` so it can be serialized into
	LLMErrorResponse by the central ExceptionHandler.
	"""

	error_code: str = "internal_error"
	status: int = 500
	reason: str = "Internal agent error"

	def __init__(self, message: str, details: dict[str, Any] | None = None):
		self.message = message
		self.details = details or {}
		super().__init__(message)


class LLMGatewayException(AgentException):
	"""Generic gateway failure (network, 5xx, unexpected SDK error).

	The provider/model could not be reached, so from the caller's perspective the
	model is inaccessible.
	"""

	error_code = "model_is_inaccessible"
	status = 503
	reason = "Model is inaccessible"


class AuthenticationException(AgentException):
	"""Bad/expired API key, 401/403 from provider."""

	error_code = "provider_auth_failed"
	status = 401
	reason = "Provider authentication failed"


class RateLimitException(AgentException):
	"""429 / quota exhausted."""

	error_code = "rate_limit_exceeded"
	status = 429
	reason = "Rate limit exceeded"


class ContentSafetyException(AgentException):
	"""Response blocked by provider safety filters."""

	error_code = "content_safety_blocked"
	status = 422
	reason = "Response blocked by safety filter"


class JSONParsingException(AgentException):
	"""Provider returned non-JSON when JSON was expected."""

	error_code = "response_is_invalid"
	status = 502
	reason = "Provider returned an invalid response"


class UnknownModelException(AgentException):
	"""Model id not in MODEL_PROVIDER_MAP / not supported."""

	error_code = "model_is_unknown"
	status = 404
	reason = "Model is not supported"
