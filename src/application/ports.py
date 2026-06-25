import abc
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from src.domain.models import ChatRoles, LLMModelType, LLMProvider
from src.domain.chat_settings import ChatSettings


# --- Shared DTOs ---------------------------------------------------------


class UserMessageDTO(BaseModel):
	"""A single conversation turn received from / returned to the backend."""

	model_config = ConfigDict(frozen=True)

	chat_id: UUID
	message: str
	llm_model: LLMModelType | None = LLMModelType.testing_mock
	role: ChatRoles


class LLMResponse(BaseModel):
	"""Raw provider generation result (provider-internal, used for logging/cost)."""

	text: str
	model: LLMModelType
	usage: dict | None = None
	provider: str


class LLMRequest(BaseModel):
	"""RabbitMQ request envelope: the current message plus prior history.

	History is ordered oldest -> newest. The `message` field is the newest
	user turn and is appended by the chosen gateway.
	"""

	model_config = ConfigDict(frozen=True)

	message: UserMessageDTO
	history: list[UserMessageDTO] = []
	chat_settings: ChatSettings | None = None


class LLMErrorResponse(BaseModel):
	"""Structured failure returned to the backend."""

	error_code: str  # machine-readable snake_case, e.g. "model_is_inaccessible"
	status: int  # HTTP-style status, e.g. 503
	reason: str  # short human-readable reason phrase
	message: str  # detailed message
	provider: str | None = None
	details: dict = {}


class LLMResult(BaseModel):
	"""Wire RESULT envelope published to llm.agent.result.

	The backend correlates by `chat_id` (and the propagated correlation_id).
	Exactly one of `message` / `error` is set.
	"""

	chat_id: UUID
	message: UserMessageDTO | None = None  # the model's reply (role=MODEL) on success
	error: LLMErrorResponse | None = None  # set on failure


# --- Gateway port: one raw external API call -----------------------------


class ILLMProviderGateway(abc.ABC):
	"""Raw transport to a single LLM vendor.

	Raises AgentException subclasses on failure; returns LLMResponse on success.
	"""

	provider: LLMProvider  # set by each concrete gateway as a class attr

	@abc.abstractmethod
	async def generate(
		self,
		model: LLMModelType,
		system_prompt: str,
		user_message: str,
		history: list[UserMessageDTO],
		chat_settings: ChatSettings | None = None,
	) -> LLMResponse: ...


# --- Service port: provider-agnostic business wrapper -------------------


class ILLMProviderService(abc.ABC):
	"""Wraps a gateway with provider-specific message mapping.

	Returns a UserMessageDTO with role=MODEL — the assistant's reply.
	"""

	@abc.abstractmethod
	async def generate(self, request: LLMRequest) -> UserMessageDTO: ...


# --- Dispatcher port ----------------------------------------------------


class IAgentService(abc.ABC):
	"""Top-level entry point. Routes `request.message.llm_model` to the
	correct provider service via MODEL_PROVIDER_MAP.
	"""

	@abc.abstractmethod
	async def handle(self, request: LLMRequest) -> UserMessageDTO: ...
