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
