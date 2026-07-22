from dataclasses import dataclass
from logging import Logger
from typing import ClassVar

from anthropic import APIError, AsyncAnthropic

from src.application.ports import ILLMProviderGateway, LLMResponse, UserMessageDTO
from src.conf import settings
from src.domain.chat_settings import ChatSettings, resolve_max_tokens, resolve_temperature
from src.domain.models import ChatRoles, LLMModelType, LLMProvider
from src.infrastructure.exception_handler import ExceptionHandler
from src.infrastructure.exceptions import ContentSafetyException


def _to_anthropic_messages(user_message: str, history: list[UserMessageDTO]) -> list[dict]:
	"""Map our ChatRoles to Anthropic roles. SYSTEM turns are pulled out by the
	caller and passed as the top-level `system` param, so they never appear here.
	"""
	messages: list[dict] = []
	for m in history:
		if m.role == ChatRoles.SYSTEM:
			continue  # handled by system_prompt
		role = "assistant" if m.role == ChatRoles.MODEL else "user"
		messages.append({"role": role, "content": m.message})
	messages.append({"role": "user", "content": user_message})
	return messages


@dataclass
class AnthropicGateway(ILLMProviderGateway):
	provider: ClassVar[LLMProvider] = LLMProvider.ANTHROPIC

	logger: Logger
	_client: AsyncAnthropic | None = None

	async def generate(
		self,
		model: LLMModelType,
		system_prompt: str,
		user_message: str,
		history: list[UserMessageDTO],
		chat_settings: ChatSettings | None = None,
	) -> LLMResponse:
		if self._client is None:
			self._client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
		try:
			resp = await self._client.messages.create(
				model=model.value,
				system=system_prompt,
				messages=_to_anthropic_messages(user_message, history),
				max_tokens=resolve_max_tokens(chat_settings),
				temperature=resolve_temperature(chat_settings),
			)
		except APIError as e:
			raise ExceptionHandler.classify_provider_error(
				e,
				provider="Anthropic",
				status=getattr(e, "status_code", None),
				body=getattr(e, "body", None),
			) from e

		stop_reason = getattr(resp, "stop_reason", None)
		if stop_reason == "content_filtered":
			raise ContentSafetyException(
				message="Anthropic filtered the response",
				details={"stop_reason": stop_reason},
			)

		text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
		usage = {
			"input_tokens": resp.usage.input_tokens,
			"output_tokens": resp.usage.output_tokens,
		}
		return LLMResponse(text=text, model=model, usage=usage, provider=LLMProvider.ANTHROPIC.value)
