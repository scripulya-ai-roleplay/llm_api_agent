### Scripulya Agent

`scripulya_agent` is a FastStream/RabbitMQ microservice that acts as a **pure LLM-provider agent**. It consumes chat requests (a new message + history) from RabbitMQ, calls the requested LLM provider (Anthropic, Google, Z.ai/GLM, DeepSeek, or a built-in Mock), and **publishes the assistant's reply to a result queue**.

It is intentionally lightweight: **no database, no HTTP server, no auth** (auth is handled by the backend). It is structured after the `scripulya_ai` Clean / Hexagonal template (`domain → application → infrastructure → controllers`) and uses `dishka` for dependency injection.

---

### How it talks to the backend

Communication is **asynchronous** over two RabbitMQ queues:

| Queue | Direction | Payload |
|-------|-----------|---------|
| `llm.agent.request` | backend → agent | `LLMRequest` |
| `llm.agent.result`  | agent → backend  | `LLMResult` |

The backend **publishes** an `LLMRequest` (fire-and-forget) and **consumes** `llm.agent.result`, correlating each result back to its request by `chat_id` (and the RabbitMQ `correlation_id`, which the service propagates to the result message).

#### `LLMRequest`

```jsonc
{
  "message": {                      // UserMessageDTO — the new turn
    "chat_id": "uuid",
    "message": "user text",
    "llm_model": "claude-sonnet-4-20250514",   // see LLMModelType; defaults to "testing_mock"
    "role": "user"
  },
  "history": [                       // list[UserMessageDTO] — prior turns, oldest → newest
    { "chat_id": "uuid", "message": "...", "llm_model": "...", "role": "user" },
    { "chat_id": "uuid", "message": "...", "llm_model": "...", "role": "model" }
  ]
}
```

#### `LLMResult`

```jsonc
{
  "chat_id": "uuid",                 // matches the request's chat_id
  "message": {                       // UserMessageDTO | null — present on success
    "chat_id": "uuid",
    "message": "assistant reply text",
    "llm_model": "claude-sonnet-4-20250514",
    "role": "model"
  },
  "error": null                      // LLMErrorResponse | null — present on failure
}
```

On failure, `message` is `null` and `error` is:

```jsonc
{
  "error_code": "provider_auth_failed",      // machine-readable snake_case
  "status": 401,                             // HTTP-style status
  "reason": "Provider authentication failed", // short human-readable phrase
  "message": "...",                          // detailed message
  "provider": "anthropic",
  "details": {}
}
```

**Backend handling:**

1. Publish `LLMRequest` to `llm.agent.request` (set a `correlation_id`).
2. Consume `llm.agent.result`; correlate by `chat_id`.
3. If `error` is set, branch on `error_code` (each carries an HTTP-style `status`):
   - `provider_auth_failed` (401) / `rate_limit_exceeded` (429) → retry / surface "try later"
   - `content_safety_blocked` (422) → moderation message
   - `model_is_unknown` (404) / `response_is_invalid` (502) → client / upstream error
   - `model_is_inaccessible` (503) / `internal_error` (500) → log + alert (5xx-class)
4. Otherwise persist `message` (a `UserMessageDTO` with `role=model`) against `chat_id`.

#### Supported models (`LLMModelType`)

| Provider | Models |
|----------|--------|
| Mock | `testing_mock` |
| Anthropic | `claude-sonnet-4-20250514`, `claude-haiku-4-20250514` |
| Google | `gemini-3-flash-preview`, `gemini-2.5-pro` |
| Z.ai (GLM) | `glm-4.6`, `glm-4.5` |
| DeepSeek | `deepseek-chat`, `deepseek-reasoner` |

The model → provider mapping is the single source of truth in `src/domain/models.py` (`MODEL_PROVIDER_MAP`).

---

### Project Structure

```
scripulya_agent/
├── src/
│   ├── main.py                 # entrypoint: logging + asyncio.run(app.run())
│   ├── app.py                  # FastStream app, RabbitBroker, setup_dishka, lifespan
│   ├── conf.py                 # Settings (pydantic-settings) + settings singleton
│   ├── domain/models.py        # ChatRoles, LLMProvider, LLMModelType, MODEL_PROVIDER_MAP
│   ├── application/
│   │   ├── ports.py            # DTOs + ILLMProviderGateway/ILLMProviderService/IAgentService
│   │   ├── agent/service.py    # AgentService: routes model → provider service
│   │   └── {anthropic,google,zai,deepseek,mock}/service.py
│   ├── infrastructure/
│   │   ├── exceptions.py          # AgentException hierarchy (carries error_code/status/reason)
│   │   ├── exception_handler.py   # ExceptionHandler: maps any exception -> LLMErrorResponse
│   │   ├── di.py                  # dishka providers + create_container()
│   │   ├── logging/trace.py       # correlation-id ContextVar + logging Filter
│   │   └── gateways/{anthropic,google,zai,deepseek,mock}_gateway.py
│   └── controllers/llm.py      # RabbitRouter: subscriber(request) + publisher(result)
├── tests/{unit,e2e}/
├── docker-compose.yml          # app + rabbitmq:3-management
├── Dockerfile                  # python:3.13-slim, CMD python -m src.main
└── pyproject.toml              # deps + Ruff config (tabs, double quotes, 120)
```

---

### Local Development

#### Prerequisites
- Docker + Docker Compose (recommended)
- Python 3.13 (for running tests/linters outside Docker)

#### Run with Docker Compose

```bash
cp .env.example .env          # fill in provider keys (or leave empty to use the Mock)
docker compose up --build
```

Starts `app` + `rabbitmq:3-management` (broker on `:5672`, UI on `http://localhost:15672`, `guest/guest`).

#### Run without Docker

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
RABBIT_URL=amqp://guest:guest@localhost:5672/ python -m src.main
```

---

### Linters & Formatting

[Ruff](https://docs.astral.sh/ruff/) (lint + format) + [pre-commit](https://pre-commit.com/). Config: line-length 120, tab indent, double quotes (see `pyproject.toml`).

```bash
source .venv/bin/activate
pip install pre-commit ruff
pre-commit install
ruff check .
ruff format .
```

---

### Running Tests

```bash
pytest                        # all
pytest -m unit                # unit only (no broker needed)
pytest -m e2e                 # in-memory broker flow (no RabbitMQ required)
```

---

### CI

`.github/workflows/main.yml` runs Ruff (lint + format check), unit tests, and e2e tests (against a `rabbitmq:3-management` service) on push/PR.
