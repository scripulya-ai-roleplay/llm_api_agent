import logging

import pytest

from src.infrastructure.exception_handler import ExceptionHandler
from src.infrastructure.exceptions import (
	AgentException,
	AuthenticationException,
	ContentSafetyException,
	JSONParsingException,
	LLMGatewayException,
	RateLimitException,
	UnknownModelException,
)


@pytest.fixture
def handler() -> ExceptionHandler:
	return ExceptionHandler(logger=logging.getLogger("test"))


@pytest.mark.unit
class TestExceptionHandler:
	@pytest.mark.parametrize(
		("exc", "error_code", "status", "reason"),
		[
			(LLMGatewayException("boom"), "model_is_inaccessible", 503, "Model is inaccessible"),
			(AuthenticationException("no key"), "provider_auth_failed", 401, "Provider authentication failed"),
			(RateLimitException("slow down"), "rate_limit_exceeded", 429, "Rate limit exceeded"),
			(ContentSafetyException("blocked"), "content_safety_blocked", 422, "Response blocked by safety filter"),
			(JSONParsingException("bad json"), "response_is_invalid", 502, "Provider returned an invalid response"),
			(UnknownModelException("nope"), "model_is_unknown", 404, "Model is not supported"),
		],
	)
	def test_agent_exceptions_map_to_structured_error(self, handler, exc, error_code, status, reason):
		response = handler.handle(exc, provider="anthropic")

		assert response.error_code == error_code
		assert response.status == status
		assert response.reason == reason
		assert response.message == exc.message
		assert response.provider == "anthropic"
		assert response.details == exc.details

	def test_carries_message_and_details(self, handler):
		exc = RateLimitException("quota exhausted", details={"retry_after": 30})

		response = handler.handle(exc)

		assert response.message == "quota exhausted"
		assert response.details == {"retry_after": 30}
		assert response.provider is None

	def test_base_agent_exception_falls_back_to_internal_error(self, handler):
		response = handler.handle(AgentException("oops"))

		assert response.error_code == "internal_error"
		assert response.status == 500
		assert response.reason == "Internal agent error"
		assert response.message == "oops"

	def test_unexpected_exception_surfaces_real_cause(self, handler):
		# A non-domain exception must surface its real message/code (here a generic
		# gateway error) instead of being masked as a generic "Internal agent error".
		response = handler.handle(RuntimeError("kaboom"), provider="google")

		assert response.error_code == "model_is_inaccessible"
		assert response.status == 503
		assert response.reason == "Model is inaccessible"
		assert response.message == "google gateway error: kaboom"
		assert response.provider == "google"

	def test_missing_api_key_surfaces_as_auth_error(self, handler):
		# The google-genai SDK raises a ValueError (no HTTP status) when the API key
		# is empty; it must be classified as an auth failure, not hidden.
		response = handler.handle(ValueError("No API key was provided"), provider="google")

		assert response.error_code == "provider_auth_failed"
		assert response.status == 401
		assert "No API key was provided" in response.message
		assert response.provider == "google"


@pytest.mark.unit
class TestClassifyProviderError:
	@pytest.mark.parametrize(
		("status", "expected", "message"),
		[
			(401, AuthenticationException, "DeepSeek auth failed (401): boom"),
			(403, AuthenticationException, "DeepSeek auth failed (403): boom"),
			(429, RateLimitException, "DeepSeek rate limit / quota exceeded: boom"),
			(500, LLMGatewayException, "DeepSeek API error (500): boom"),
		],
	)
	def test_known_status_maps_to_exception(self, status, expected, message):
		exc = ExceptionHandler.classify_provider_error(RuntimeError("boom"), provider="DeepSeek", status=status)

		assert isinstance(exc, expected)
		assert exc.message == message
		assert exc.details["status"] == status
		assert exc.details["error_type"] == "RuntimeError"

	@pytest.mark.parametrize(
		("message", "expected"),
		[
			("401 Permission denied", AuthenticationException),
			("403 Forbidden", AuthenticationException),
			("No API key was provided", AuthenticationException),
			("429 resource_exhausted", RateLimitException),
			("quota exceeded", RateLimitException),
		],
	)
	def test_infers_status_from_message_when_missing(self, message, expected):
		exc = ExceptionHandler.classify_provider_error(RuntimeError(message), provider="Google")

		assert isinstance(exc, expected)

	def test_generic_error_without_status_is_gateway_error(self):
		exc = ExceptionHandler.classify_provider_error(RuntimeError("something broke"), provider="Google")

		assert isinstance(exc, LLMGatewayException)
		assert exc.message == "Google gateway error: something broke"
		assert exc.details["status"] is None
		assert "body" not in exc.details

	def test_body_included_in_details_when_present(self):
		exc = ExceptionHandler.classify_provider_error(
			RuntimeError("boom"), provider="Z.ai", status=500, body={"error": "x"}
		)

		assert exc.details["body"] == str({"error": "x"})
