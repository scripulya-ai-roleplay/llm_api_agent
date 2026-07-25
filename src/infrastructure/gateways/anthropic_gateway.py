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
from src.infrastructure.gateways._streaming import emit_token


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
		on_token=None,
	) -> LLMResponse:
		if self._client is None:
			self._client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
		try:
			async with self._client.messages.stream(
				model=model.value,
				system=system_prompt,
				messages=_to_anthropic_messages(user_message, history),
				max_tokens=resolve_max_tokens(chat_settings),
				temperature=resolve_temperature(chat_settings),
			) as stream:
				async for delta in stream.text_stream:
					if delta:
						await emit_token(on_token, delta)
				final = await stream.get_final_message()
		except APIError as e:
			raise ExceptionHandler.classify_provider_error(
				e,
				provider="Anthropic",
				status=getattr(e, "status_code", None),
				body=getattr(e, "body", None),
			) from e

		stop_reason = getattr(final, "stop_reason", None)
		if stop_reason == "content_filtered":
			raise ContentSafetyException(
				message="Anthropic filtered the response",
				details={"stop_reason": stop_reason},
			)

		text = "".join(b.text for b in final.content if getattr(b, "type", None) == "text")
		usage = {
			"input_tokens": final.usage.input_tokens,
			"output_tokens": final.usage.output_tokens,
		}
		return LLMResponse(text=text, model=model, usage=usage, provider=LLMProvider.ANTHROPIC.value)
