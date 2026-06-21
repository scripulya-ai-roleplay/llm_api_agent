from enum import StrEnum


class ChatRoles(StrEnum):
	"""Roles in a conversation. MODEL is the assistant's reply."""

	SYSTEM = "system"
	USER = "user"
	MODEL = "model"


class LLMProvider(StrEnum):
	"""Which external LLM vendor serves a request."""

	MOCK = "mock"
	ANTHROPIC = "anthropic"
	GOOGLE = "google"
	ZAI = "zai"
	DEEPSEEK = "deepseek"


class LLMModelType(StrEnum):
	"""Concrete model identifiers as accepted by each provider's API.

	The string value is the exact model id passed to the provider SDK.
	"""

	testing_mock = "testing_mock"

	# Anthropic
	claude_sonnet = "claude-sonnet-4-20250514"
	claude_haiku = "claude-haiku-4-20250514"

	# Google
	gemini_flash_preview = "gemini-3-flash-preview"
	gemini_pro = "gemini-2.5-pro"

	# Z.ai (GLM, OpenAI-compatible)
	glm_5_2 = "glm-5.2"
	glm_4_6 = "glm-4.6"
	glm_4_5 = "glm-4.5"

	# DeepSeek (OpenAI-compatible)
	deepseek_chat = "deepseek-chat"
	deepseek_reasoner = "deepseek-reasoner"


# Authoritative model -> provider routing table.
# Single source of truth used by the AgentService dispatcher.
MODEL_PROVIDER_MAP: dict[LLMModelType, LLMProvider] = {
	LLMModelType.testing_mock: LLMProvider.MOCK,
	LLMModelType.claude_sonnet: LLMProvider.ANTHROPIC,
	LLMModelType.claude_haiku: LLMProvider.ANTHROPIC,
	LLMModelType.gemini_flash_preview: LLMProvider.GOOGLE,
	LLMModelType.gemini_pro: LLMProvider.GOOGLE,
	LLMModelType.glm_5_2: LLMProvider.ZAI,
	LLMModelType.glm_4_6: LLMProvider.ZAI,
	LLMModelType.glm_4_5: LLMProvider.ZAI,
	LLMModelType.deepseek_chat: LLMProvider.DEEPSEEK,
	LLMModelType.deepseek_reasoner: LLMProvider.DEEPSEEK,
}
