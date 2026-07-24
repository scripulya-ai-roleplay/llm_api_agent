import asyncio
import json
from uuid import uuid4

import pytest
from dishka.integrations.faststream import setup_dishka
from faststream.rabbit import TestRabbitBroker

from src.app import create_broker
from src.application.ports import LLMRequest, LLMResult, UserMessageDTO
from src.conf import settings
from src.controllers.llm import router as llm_router
from src.domain.chat_settings import (
	ChatSettings,
	ControlBehavior,
	FunctionsSettings,
	Perspective,
	Preset,
	ReasoningEffort,
	ResponseLength,
	TemperatureSettings,
	TokenLimit,
	Toggle,
)
from src.domain.models import ChatRoles, LLMModelType
from src.infrastructure.di import create_container
from src.infrastructure.gateways.mock_gateway import MockGateway


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_mock_request_publishes_model_reply():
	"""End-to-end (in-memory broker): a mock request flows subscriber -> agent
	-> mock provider and an LLMResult with the model reply is published.
	"""
	broker = create_broker()
	broker.include_router(llm_router)

	captured: list[LLMResult] = []

	@broker.subscriber(settings.LLM_RESULT_QUEUE)
	async def spy(result: LLMResult) -> None:
		captured.append(result)

	# Real container; the mock path needs no provider keys.
	container = create_container()
	setup_dishka(container=container, broker=broker, auto_inject=True)

	chat_id = uuid4()
	request = LLMRequest(
		message=UserMessageDTO(
			chat_id=chat_id,
			message="hello agent",
			llm_model=LLMModelType.testing_mock,
			role=ChatRoles.USER,
		)
	)

	async with TestRabbitBroker(broker) as tb:
		await tb.publish(request.model_dump(mode="json"), settings.LLM_REQUEST_QUEUE)

	assert len(captured) == 1
	result = captured[0]
	assert result.chat_id == chat_id
	assert result.error is None
	assert result.message is not None
	assert result.message.role == ChatRoles.MODEL
	assert result.message.message == "Mock response for: hello agent"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_request_with_chat_settings_is_accepted():
	"""The LLMRequest contract carries a ChatSettings payload and still deserializes
	and flows subscriber -> agent -> mock provider, proving the new field is
	accepted end-to-end (the mock provider ignores the settings).
	"""
	broker = create_broker()
	broker.include_router(llm_router)

	captured: list[LLMResult] = []

	@broker.subscriber(settings.LLM_RESULT_QUEUE)
	async def spy(result: LLMResult) -> None:
		captured.append(result)

	container = create_container()
	setup_dishka(container=container, broker=broker, auto_inject=True)

	chat_id = uuid4()
	chat_settings = ChatSettings(
		aiControlBehavior=ControlBehavior.CONTROL,
		continueBehavior=ControlBehavior.DONT_CONTROL,
		perspective=Perspective.SECOND_PERSON,
		temperature=TemperatureSettings(preset=Preset.HIGH, value=0.42),
		responseLength=ResponseLength.LONG,
		responseTokenLimit=TokenLimit.HIGH,
		reasoning=Toggle.ON,
		reasoningEffort=ReasoningEffort.MID,
		aiMediaPicker=Toggle.OFF,
		functions=FunctionsSettings(characterNameGenerator=True),
	)
	request = LLMRequest(
		message=UserMessageDTO(
			chat_id=chat_id,
			message="hello with settings",
			llm_model=LLMModelType.testing_mock,
			role=ChatRoles.USER,
		),
		chat_settings=chat_settings,
	)

	async with TestRabbitBroker(broker) as tb:
		await tb.publish(request.model_dump(mode="json"), settings.LLM_REQUEST_QUEUE)

	# Round-trip: the payload (with chat_settings) deserialized on the subscriber side.
	assert len(captured) == 1
	result = captured[0]
	assert result.chat_id == chat_id
	assert result.error is None
	assert result.message is not None
	assert result.message.message == "Mock response for: hello with settings"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_handler_marks_generation_done_with_correlation_id(monkeypatch):
	"""The handler marks each generation done (on success) using the backend's
	correlation_id as the request_id, so the backend watchdog can drain it instead
	of declaring it dead once the heartbeat TTL lapses."""
	marked: list[str] = []

	async def fake_mark_done(redis_client, request_id):  # noqa: ANN001
		marked.append(request_id)

	class _NoopHeartbeat:
		def __init__(self, *args, **kwargs):
			pass

		async def __aenter__(self):
			return self

		async def __aexit__(self, *exc):
			return False

	# Heartbeat/mark_done are patched to no-ops so the test doesn't need a live Redis;
	# the real Redis round-trip is covered by the deploy-level manual check.
	monkeypatch.setattr("src.controllers.llm.Heartbeat", _NoopHeartbeat)
	monkeypatch.setattr("src.controllers.llm.mark_done", fake_mark_done)

	broker = create_broker()
	broker.include_router(llm_router)

	captured: list[LLMResult] = []

	@broker.subscriber(settings.LLM_RESULT_QUEUE)
	async def spy(result: LLMResult) -> None:
		captured.append(result)

	container = create_container()
	setup_dishka(container=container, broker=broker, auto_inject=True)

	chat_id = uuid4()
	request = LLMRequest(
		message=UserMessageDTO(
			chat_id=chat_id,
			message="hello agent",
			llm_model=LLMModelType.testing_mock,
			role=ChatRoles.USER,
		)
	)

	async with TestRabbitBroker(broker) as tb:
		await tb.publish(request.model_dump(mode="json"), settings.LLM_REQUEST_QUEUE, correlation_id="rid-xyz")

	assert len(captured) == 1
	assert captured[0].error is None
	assert marked == ["rid-xyz"]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_generation_timeout_returns_structured_error(monkeypatch):
	"""A provider call exceeding LLM_GENERATION_TIMEOUT_SECONDS is cancelled and
	returned as a generation_timeout error rather than left hanging indefinitely."""

	async def hang(self, *args, **kwargs):  # noqa: ANN001, ARG002
		await asyncio.sleep(30)

	monkeypatch.setattr(MockGateway, "generate", hang)
	monkeypatch.setattr(settings, "LLM_GENERATION_TIMEOUT_SECONDS", 0.1)

	# No live Redis: neutralize the heartbeat path (covered by unit tests).
	class _NoopHeartbeat:
		def __init__(self, *args, **kwargs):
			pass

		async def __aenter__(self):
			return self

		async def __aexit__(self, *exc):
			return False

	async def _noop_mark_done(redis_client, request_id):  # noqa: ANN001, ARG001
		return None

	monkeypatch.setattr("src.controllers.llm.Heartbeat", _NoopHeartbeat)
	monkeypatch.setattr("src.controllers.llm.mark_done", _noop_mark_done)

	broker = create_broker()
	broker.include_router(llm_router)

	captured: list[LLMResult] = []

	@broker.subscriber(settings.LLM_RESULT_QUEUE)
	async def spy(result: LLMResult) -> None:
		captured.append(result)

	container = create_container()
	setup_dishka(container=container, broker=broker, auto_inject=True)

	chat_id = uuid4()
	request = LLMRequest(
		message=UserMessageDTO(
			chat_id=chat_id,
			message="hang please",
			llm_model=LLMModelType.testing_mock,
			role=ChatRoles.USER,
		)
	)

	async with TestRabbitBroker(broker) as tb:
		await tb.publish(request.model_dump(mode="json"), settings.LLM_REQUEST_QUEUE, correlation_id="rid-timeout")

	assert len(captured) == 1
	err = captured[0].error
	assert err is not None
	assert err.error_code == "generation_timeout"
	assert err.status == 504
	assert "0.1s" in err.message


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_mock_request_streams_tokens_then_terminal_frame(monkeypatch):
	"""Per-token frames and a terminal 'done' frame are published to the
	gen:{request_id}:tokens Redis channel, keyed by the backend's correlation_id.

	A fake Redis (no live instance) records publishes; the heartbeat/mark_done paths
	are neutralized as in the other e2e tests.
	"""
	published: list[tuple[str, dict]] = []

	class _FakeRedis:
		async def publish(self, channel, payload):
			published.append((channel, json.loads(payload)))

		async def aclose(self):
			return None

	monkeypatch.setattr("redis.asyncio.from_url", lambda *args, **kwargs: _FakeRedis())

	class _NoopHeartbeat:
		def __init__(self, *args, **kwargs):
			pass

		async def __aenter__(self):
			return self

		async def __aexit__(self, *exc):
			return False

	async def _noop_mark_done(redis_client, request_id):  # noqa: ANN001, ARG001
		return None

	monkeypatch.setattr("src.controllers.llm.Heartbeat", _NoopHeartbeat)
	monkeypatch.setattr("src.controllers.llm.mark_done", _noop_mark_done)

	broker = create_broker()
	broker.include_router(llm_router)

	captured: list[LLMResult] = []

	@broker.subscriber(settings.LLM_RESULT_QUEUE)
	async def spy(result: LLMResult) -> None:
		captured.append(result)

	container = create_container()
	setup_dishka(container=container, broker=broker, auto_inject=True)

	chat_id = uuid4()
	request = LLMRequest(
		message=UserMessageDTO(
			chat_id=chat_id,
			message="hello agent",
			llm_model=LLMModelType.testing_mock,
			role=ChatRoles.USER,
		)
	)

	async with TestRabbitBroker(broker) as tb:
		await tb.publish(request.model_dump(mode="json"), settings.LLM_REQUEST_QUEUE, correlation_id="rid-stream")

	# All frames land on the request-scoped channel.
	assert {channel for channel, _ in published} == {"gen:rid-stream:tokens"}

	token_text = "".join(frame["text"] for _, frame in published if frame.get("type") == "token")
	assert token_text == "Mock response for: hello agent"

	# Exactly one terminal frame, and it follows the tokens.
	terminals = [frame for _, frame in published if frame.get("type") in ("done", "error")]
	assert terminals == [{"type": "done"}]
	# Generation still completed normally via the result queue.
	assert len(captured) == 1
	assert captured[0].error is None
