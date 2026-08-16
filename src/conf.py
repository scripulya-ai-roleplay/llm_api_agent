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

	# --- Redis heartbeat (anti-hang) ---
	# LLM_HEARTBEAT_ALIVE_TTL MUST match the backend's, else the watchdog mistimes.
	REDIS_URL: str = "redis://redis:6379/0"
	LLM_HEARTBEAT_ALIVE_TTL: int = 30
	LLM_HEARTBEAT_REFRESH_INTERVAL_SECONDS: int = 10

	# --- Provider API keys (empty = provider disabled at runtime, service still boots) ---
	ANTHROPIC_API_KEY: str = ""
	GEMINI_API_KEY: str = ""
	ZAI_API_KEY: str = ""
	ZAI_BASE_URL: str = "https://api.z.ai/api/paas/v4"
	QWEN_API_KEY: str = ""
	QWEN_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
	DEEPSEEK_API_KEY: str = ""
	DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"

	# --- Generation defaults ---
	LLM_TEMPERATURE: float = 0.7
	LLM_MAX_TOKENS: int = 4096
	# Hard cap on a single provider call. A hung SDK call (the agent stays alive but
	# the model never returns) is not something the heartbeat can detect, so this
	# turns it into a fast generation_timeout failure instead of an indefinite wait.
	LLM_GENERATION_TIMEOUT_SECONDS: int = 300

	# Single shared system prompt; the backend may override per-message later.
	SYSTEM_PROMPT: str = "You are a helpful assistant."


settings = Settings()  # type: ignore[call-arg]
