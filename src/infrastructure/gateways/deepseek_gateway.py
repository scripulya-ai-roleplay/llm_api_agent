from dataclasses import dataclass
from logging import Logger
from typing import ClassVar

from openai import APIError, AsyncOpenAI

from src.application.ports import ILLMProviderGateway, LLMResponse, UserMessageDTO
from src.conf import settings
from src.domain.chat_settings import ChatSettings, resolve_max_tokens, resolve_temperature
from src.domain.models import LLMModelType, LLMProvider
from src.infrastructure.exception_handler import ExceptionHandler
from src.infrastructure.exceptions import ContentSafetyException
from src.infrastructure.gateways.zai_gateway import _to_openai_messages


@dataclass
class DeepSeekGateway(ILLMProviderGateway):
	"""DeepSeek — OpenAI-compatible API."""

	provider: ClassVar[LLMProvider] = LLMProvider.DEEPSEEK

	logger: Logger
	_client: AsyncOpenAI | None = None

	async def generate(
		self,
		model: LLMModelType,
		system_prompt: str,
		user_message: str,
		history: list[UserMessageDTO],
		chat_settings: ChatSettings | None = None,
	) -> LLMResponse:
		if self._client is None:
			self._client = AsyncOpenAI(api_key=settings.DEEPSEEK_API_KEY, base_url=settings.DEEPSEEK_BASE_URL)
		messages = _to_openai_messages(system_prompt, user_message, history)
		try:
			resp = await self._client.chat.completions.create(
				model=model.value,
				messages=messages,
				temperature=resolve_temperature(chat_settings),
				max_tokens=resolve_max_tokens(chat_settings),
			)
		except APIError as e:
			raise ExceptionHandler.classify_provider_error(
				e,
				provider="DeepSeek",
				status=getattr(e, "status_code", None),
				body=getattr(e, "body", None),
			) from e

		choice = resp.choices[0]
		if choice.finish_reason == "content_filter":
			raise ContentSafetyException(message="DeepSeek filtered the response", details={})

		usage = (
			{"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}
			if resp.usage
			else None
		)
		return LLMResponse(
			text=choice.message.content or "",
			model=model,
			usage=usage,
			provider=LLMProvider.DEEPSEEK.value,
		)
