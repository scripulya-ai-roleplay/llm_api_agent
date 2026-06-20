import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
import pytest
from openai import APIStatusError

from src.application.ports import UserMessageDTO
from src.domain.models import ChatRoles, LLMModelType
from src.infrastructure.exceptions import AuthenticationException
from src.infrastructure.gateways.zai_gateway import ZaiGateway


def _status_error(status_code: int) -> APIStatusError:
	request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
	response = httpx.Response(status_code=status_code, request=request)
	return APIStatusError(message=f"err {status_code}", response=response, body={})


def _fake_response(content="zai reply", finish_reason="stop") -> SimpleNamespace:
	return SimpleNamespace(
		choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)],
		usage=SimpleNamespace(prompt_tokens=5, completion_tokens=9),
	)


@pytest.mark.unit
class TestZaiGateway:
	@pytest.fixture
	def gateway(self) -> ZaiGateway:
		client = MagicMock()
		client.chat.completions.create = AsyncMock(return_value=_fake_response())
		return ZaiGateway(logger=logging.getLogger(), _client=client)

	@pytest.mark.asyncio
	async def test_success_extracts_text_and_usage(self, gateway):
		result = await gateway.generate(
			model=LLMModelType.glm_4_6,
			system_prompt="sys",
			user_message="hi",
			history=[],
		)
		assert result.text == "zai reply"
		assert result.provider == "zai"
		assert result.usage == {"prompt_tokens": 5, "completion_tokens": 9}

		kwargs = gateway._client.chat.completions.create.call_args.kwargs
		assert kwargs["model"] == "glm-4.6"
		# first message is the system prompt, last is the user message
		assert kwargs["messages"][0] == {"role": "system", "content": "sys"}
		assert kwargs["messages"][-1] == {"role": "user", "content": "hi"}

	@pytest.mark.asyncio
	async def test_history_roles_mapped(self, gateway):
		history = [
			UserMessageDTO(chat_id=uuid4(), message="u1", llm_model=LLMModelType.glm_4_6, role=ChatRoles.USER),
			UserMessageDTO(chat_id=uuid4(), message="m1", llm_model=LLMModelType.glm_4_6, role=ChatRoles.MODEL),
		]
		await gateway.generate(
			model=LLMModelType.glm_4_6,
			system_prompt="sys",
			user_message="u2",
			history=history,
		)
		messages = gateway._client.chat.completions.create.call_args.kwargs["messages"]
		roles = [m["role"] for m in messages]
		assert roles == ["system", "user", "assistant", "user"]

	@pytest.mark.asyncio
	async def test_auth_error_maps_to_authentication_exception(self, gateway):
		gateway._client.chat.completions.create.side_effect = _status_error(401)
		with pytest.raises(AuthenticationException):
			await gateway.generate(
				model=LLMModelType.glm_4_6,
				system_prompt="sys",
				user_message="hi",
				history=[],
			)
