from dataclasses import dataclass
from logging import Logger
from typing import ClassVar

from google import genai
from google.genai import types

from src.application.ports import ILLMProviderGateway, LLMResponse, UserMessageDTO
from src.conf import settings
from src.domain.chat_settings import (
	ChatSettings,
	reasoning_enabled,
	resolve_max_tokens,
	resolve_temperature,
	resolve_thinking_budget,
)
from src.domain.models import ChatRoles, LLMModelType, LLMProvider
from src.infrastructure.exception_handler import ExceptionHandler
from src.infrastructure.exceptions import ContentSafetyException
from src.infrastructure.gateways._streaming import emit_thinking, emit_token


def _to_gemini_contents(history: list[UserMessageDTO]) -> list[types.Content]:
	contents: list[types.Content] = []
	for m in history:
		if m.role == ChatRoles.MODEL:
			contents.append(types.ModelContent(parts=[types.Part(text=m.message)]))
		elif m.role == ChatRoles.USER:
			contents.append(types.UserContent(parts=[types.Part(text=m.message)]))
		# SYSTEM handled by config.system_instruction
	return contents


def _iter_parts(chunk) -> list[types.Part]:
	"""Safely extract the first candidate's parts from a streamed chunk.

	Returns [] for chunks that carry no content (e.g. usage-only chunks, or
	blocked candidates with empty content)."""
	candidates = getattr(chunk, "candidates", None) or []
	if not candidates:
		return []
	content = getattr(candidates[0], "content", None)
	return list(getattr(content, "parts", None) or [])


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
		on_token=None,
		on_thinking=None,
	) -> LLMResponse:
		if self._client is None:
			self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
		thinking_on = reasoning_enabled(chat_settings)
		config_kwargs: dict = {
			"system_instruction": system_prompt,
			"temperature": resolve_temperature(chat_settings),
			"max_output_tokens": resolve_max_tokens(chat_settings),
		}
		if thinking_on:
			# include_thoughts flags thought parts with part.thought=True so we can route
			# the chain-of-thought to on_thinking separately from the answer.
			config_kwargs["thinking_config"] = types.ThinkingConfig(
				include_thoughts=True,
				thinking_budget=resolve_thinking_budget(chat_settings),
			)
		config = types.GenerateContentConfig(**config_kwargs)
		parts: list[str] = []
		thinking_parts: list[str] = []
		last = None
		try:
			# generate_content_stream is `async def`, so it must be awaited to obtain
			# the async iterator (the await performs the HTTP connection setup).
			stream = await self._client.aio.models.generate_content_stream(
				model=model.value,
				contents=[*_to_gemini_contents(history), user_message],
				config=config,
			)
			async for chunk in stream:
				last = chunk
				if thinking_on:
					for part in _iter_parts(chunk):
						text = getattr(part, "text", None) or ""
						if not text:
							continue
						if getattr(part, "thought", False):
							thinking_parts.append(text)
							await emit_thinking(on_thinking, text)
						else:
							parts.append(text)
							await emit_token(on_token, text)
				else:
					text = getattr(chunk, "text", None)
					if text:
						parts.append(text)
						await emit_token(on_token, text)
		except Exception as e:  # google-genai raises ClientError / ServerError
			raise ExceptionHandler.classify_provider_error(e, provider="Google") from e

		block_reason = getattr(getattr(last, "prompt_feedback", None), "block_reason", None) if last else None
		if block_reason:
			raise ContentSafetyException(
				message=f"Google blocked the prompt: {block_reason}",
				details={"block_reason": str(block_reason)},
			)

		usage_meta = getattr(last, "usage_metadata", None) if last else None
		usage = (
			{
				"prompt_token_count": getattr(usage_meta, "prompt_token_count", 0),
				"candidates_token_count": getattr(usage_meta, "candidates_token_count", 0),
			}
			if usage_meta
			else None
		)

		return LLMResponse(
			text="".join(parts),
			model=model,
			usage=usage,
			provider=LLMProvider.GOOGLE.value,
			reasoning="".join(thinking_parts) or None,
		)
