from dataclasses import dataclass
from logging import Logger
from typing import ClassVar

from openai import APIError, AsyncOpenAI

from src.application.ports import ILLMProviderGateway, LLMResponse, UserMessageDTO
from src.conf import settings
from src.domain.chat_settings import ChatSettings, reasoning_enabled, resolve_max_tokens, resolve_temperature
from src.domain.models import ChatRoles, LLMModelType, LLMProvider
from src.infrastructure.exception_handler import ExceptionHandler
from src.infrastructure.exceptions import ContentSafetyException
from src.infrastructure.gateways._streaming import emit_thinking, emit_token


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
		chat_settings: ChatSettings | None = None,
		on_token=None,
		on_thinking=None,
	) -> LLMResponse:
		if self._client is None:
			self._client = AsyncOpenAI(api_key=settings.ZAI_API_KEY, base_url=settings.ZAI_BASE_URL)
		messages = _to_openai_messages(system_prompt, user_message, history)
		parts: list[str] = []
		reasoning_parts: list[str] = []
		finish_reason = None
		# GLM 4.5+ exposes thinking via the `thinking` param; the OpenAI SDK doesn't model
		# it, so it rides in extra_body. reasoning_content is read unconditionally so we
		# also capture thinking on models that think by default (GLM-4.6) regardless of the
		# toggle.
		extra: dict = {}
		if reasoning_enabled(chat_settings):
			extra["extra_body"] = {"thinking": {"type": "enabled"}}
		try:
			# stream_options omitted: third-party OpenAI-compatible servers may reject it,
			# so usage is unavailable on the streaming path (it was only used for logging).
			stream = await self._client.chat.completions.create(
				model=model.value,
				messages=messages,
				temperature=resolve_temperature(chat_settings),
				max_tokens=resolve_max_tokens(chat_settings),
				stream=True,
				**extra,
			)
			async for chunk in stream:
				if not chunk.choices:
					continue
				choice = chunk.choices[0]
				delta = choice.delta
				reasoning = getattr(delta, "reasoning_content", None)
				if reasoning:
					reasoning_parts.append(reasoning)
					await emit_thinking(on_thinking, reasoning)
				content = delta.content
				if content:
					parts.append(content)
					await emit_token(on_token, content)
				if choice.finish_reason is not None:
					finish_reason = choice.finish_reason
		except APIError as e:
			raise ExceptionHandler.classify_provider_error(
				e,
				provider="Z.ai",
				status=getattr(e, "status_code", None),
				body=getattr(e, "body", None),
			) from e

		if finish_reason == "content_filter":
			raise ContentSafetyException(message="Z.ai filtered the response", details={})

		return LLMResponse(
			text="".join(parts),
			model=model,
			usage=None,
			provider=LLMProvider.ZAI.value,
			reasoning="".join(reasoning_parts) or None,
		)
