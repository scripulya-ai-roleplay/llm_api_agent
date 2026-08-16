from dataclasses import dataclass
from logging import Logger
from typing import ClassVar

from anthropic import APIError, AsyncAnthropic

from src.application.ports import ILLMProviderGateway, LLMResponse, UserMessageDTO
from src.conf import settings
from src.domain.chat_settings import (
	ChatSettings,
	reasoning_enabled,
	resolve_reasoning_max_tokens,
	resolve_temperature,
	resolve_thinking_budget,
)
from src.domain.models import ChatRoles, LLMModelType, LLMProvider
from src.infrastructure.exception_handler import ExceptionHandler
from src.infrastructure.exceptions import ContentSafetyException
from src.infrastructure.gateways._streaming import emit_thinking, emit_token


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
		on_thinking=None,
	) -> LLMResponse:
		if self._client is None:
			self._client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

		thinking_on = reasoning_enabled(chat_settings)
		temperature = resolve_temperature(chat_settings)
		extra: dict = {}
		if thinking_on:
			budget = resolve_thinking_budget(chat_settings)
			extra["thinking"] = {"type": "enabled", "budget_tokens": budget}
			temperature = 1.0

		text_parts: list[str] = []
		thinking_parts: list[str] = []
		try:
			async with self._client.messages.stream(
				model=model.value,
				system=system_prompt,
				messages=_to_anthropic_messages(user_message, history),
				max_tokens=resolve_reasoning_max_tokens(chat_settings),
				temperature=temperature,
				**extra,
			) as stream:
				if thinking_on:
					await self._consume_thinking_stream(stream, on_token, on_thinking, text_parts, thinking_parts)
				else:
					async for delta in stream.text_stream:
						if delta:
							text_parts.append(delta)
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

		usage = {
			"input_tokens": final.usage.input_tokens,
			"output_tokens": final.usage.output_tokens,
		}
		return LLMResponse(
			text="".join(text_parts),
			model=model,
			usage=usage,
			provider=LLMProvider.ANTHROPIC.value,
			reasoning="".join(thinking_parts) or None,
		)

	@staticmethod
	async def _consume_thinking_stream(stream, on_token, on_thinking, text_parts, thinking_parts) -> None:
		"""Iterate the raw event stream so thinking blocks are not skipped.

		`stream.text_stream` only yields answer text; extended thinking arrives as
		separate `thinking_delta` content-block deltas that text_stream discards, so
		when thinking is on we walk the event stream and route each delta to its sink.
		"""
		async for event in stream:
			if event.type != "content_block_delta":
				continue
			delta_type = getattr(event.delta, "type", None)
			if delta_type == "thinking_delta":
				text = event.delta.thinking
				thinking_parts.append(text)
				await emit_thinking(on_thinking, text)
			elif delta_type == "text_delta":
				text = event.delta.text
				text_parts.append(text)
				await emit_token(on_token, text)
