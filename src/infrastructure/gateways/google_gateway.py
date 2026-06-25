from dataclasses import dataclass
from logging import Logger
from typing import ClassVar

from google import genai
from google.genai import types

from src.application.ports import ILLMProviderGateway, LLMResponse, UserMessageDTO
from src.conf import settings
from src.domain.chat_settings import ChatSettings, resolve_max_tokens, resolve_temperature
from src.domain.models import ChatRoles, LLMModelType, LLMProvider
from src.infrastructure.exception_handler import ExceptionHandler
from src.infrastructure.exceptions import ContentSafetyException


def _to_gemini_contents(history: list[UserMessageDTO]) -> list[types.Content]:
	contents: list[types.Content] = []
	for m in history:
		if m.role == ChatRoles.MODEL:
			contents.append(types.ModelContent(parts=[types.Part(text=m.message)]))
		elif m.role == ChatRoles.USER:
			contents.append(types.UserContent(parts=[types.Part(text=m.message)]))
		# SYSTEM handled by config.system_instruction
	return contents


@dataclass
class GoogleGateway(ILLMProviderGateway):
	provider: ClassVar[LLMProvider] = LLMProvider.GOOGLE

	logger: Logger
	_client: genai.Client | None = None

	async def generate(
		self,
		model: LLMModelType,
		system_prompt: str,
		user_message: str,
		history: list[UserMessageDTO],
		chat_settings: ChatSettings | None = None,
	) -> LLMResponse:
		if self._client is None:
			self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
		config = types.GenerateContentConfig(
			system_instruction=system_prompt,
			temperature=resolve_temperature(chat_settings),
			max_output_tokens=resolve_max_tokens(chat_settings),
		)
		try:
			resp = await self._client.aio.models.generate_content(
				model=model.value,
				contents=[*_to_gemini_contents(history), user_message],
				config=config,
			)
		except Exception as e:  # google-genai raises ClientError / ServerError
			raise ExceptionHandler.classify_provider_error(e, provider="Google") from e

		block_reason = getattr(getattr(resp, "prompt_feedback", None), "block_reason", None)
		if block_reason:
			raise ContentSafetyException(
				message=f"Google blocked the prompt: {block_reason}",
				details={"block_reason": str(block_reason)},
			)

		usage_meta = getattr(resp, "usage_metadata", None)
		usage = (
			{
				"prompt_token_count": getattr(usage_meta, "prompt_token_count", 0),
				"candidates_token_count": getattr(usage_meta, "candidates_token_count", 0),
			}
			if usage_meta
			else None
		)

		return LLMResponse(
			text=getattr(resp, "text", None) or "",
			model=model,
			usage=usage,
			provider=LLMProvider.GOOGLE.value,
		)
