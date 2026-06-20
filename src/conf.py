from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
	model_config = SettingsConfigDict(
		env_file=".env",
		env_file_encoding="utf-8",
		case_sensitive=False,
		extra="ignore",
	)

	APP_NAME: str = "scripulya-agent"
	APP_VERSION: str = "0.0.1"
	DEBUG: bool = False

	# --- RabbitMQ ---
	RABBIT_URL: str = "amqp://guest:guest@rabbitmq:5672/"
	LLM_REQUEST_QUEUE: str = "llm.agent.request"
	LLM_RESULT_QUEUE: str = "llm.agent.result"

	# --- Provider API keys (empty = provider disabled at runtime, service still boots) ---
	ANTHROPIC_API_KEY: str = ""
	GEMINI_API_KEY: str = ""
	ZAI_API_KEY: str = ""
	ZAI_BASE_URL: str = "https://api.z.ai/api/paas/v4"
	DEEPSEEK_API_KEY: str = ""
	DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"

	# --- Generation defaults ---
	LLM_TEMPERATURE: float = 0.7
	LLM_MAX_TOKENS: int = 4096

	# Single shared system prompt; the backend may override per-message later.
	SYSTEM_PROMPT: str = "You are a helpful assistant."


settings = Settings()  # type: ignore[call-arg]
