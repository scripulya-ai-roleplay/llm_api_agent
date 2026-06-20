from dataclasses import dataclass
from logging import Logger
from typing import ClassVar

from openai import APIError, AsyncOpenAI

from src.application.ports import ILLMProviderGateway, LLMResponse, UserMessageDTO
from src.conf import settings
from src.domain.models import ChatRoles, LLMModelType, LLMProvider
from src.infrastructure.exception_handler import ExceptionHandler
from src.infrastructure.exceptions import ContentSafetyException


def _to_openai_messages(system_prompt: str, user_message: str, history: list[UserMessageDTO]) -> list[dict]:
	messages: list[dict] = [{"role": "system", "content": system_prompt}]
	for m in history:
		role = "assistant" if m.role == ChatRoles.MODEL else m.role.value  # "user" / "system"
		messages.append({"role": role, "content": m.message})
	messages.append({"role": "user", "content": user_message})
	return messages


@dataclass
class ZaiGateway(ILLMProviderGateway):
	"""Z.ai (GLM) — OpenAI-compatible API."""

	provider: ClassVar[LLMProvider] = LLMProvider.ZAI

	logger: Logger
	_client: AsyncOpenAI | None = None

	async def generate(
		self,
		model: LLMModelType,
		system_prompt: str,
		user_message: str,
		history: list[UserMessageDTO],
	) -> LLMResponse:
		if self._client is None:
			self._client = AsyncOpenAI(api_key=settings.ZAI_API_KEY, base_url=settings.ZAI_BASE_URL)
		messages = _to_openai_messages(system_prompt, user_message, history)
		try:
			resp = await self._client.chat.completions.create(
				model=model.value,
				messages=messages,
				temperature=settings.LLM_TEMPERATURE,
			)
		except APIError as e:
			raise ExceptionHandler.classify_provider_error(
				e,
				provider="Z.ai",
				status=getattr(e, "status_code", None),
				body=getattr(e, "body", None),
			) from e

		choice = resp.choices[0]
		if choice.finish_reason == "content_filter":
			raise ContentSafetyException(message="Z.ai filtered the response", details={})

		usage = (
			{"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}
			if resp.usage
			else None
		)
		return LLMResponse(
			text=choice.message.content or "",
			model=model,
			usage=usage,
			provider=LLMProvider.ZAI.value,
		)
