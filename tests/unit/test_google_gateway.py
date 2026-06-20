import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.application.ports import UserMessageDTO
from src.domain.models import ChatRoles, LLMModelType
from src.infrastructure.exceptions import AuthenticationException
from src.infrastructure.gateways.google_gateway import GoogleGateway


def _fake_response(text="gemini says hi") -> SimpleNamespace:
	return SimpleNamespace(
		text=text,
		prompt_feedback=SimpleNamespace(block_reason=None),
		usage_metadata=SimpleNamespace(prompt_token_count=3, candidates_token_count=4),
	)


@pytest.mark.unit
class TestGoogleGateway:
	@pytest.fixture
	def gateway(self) -> GoogleGateway:
		client = MagicMock()
		client.aio.models.generate_content = AsyncMock(return_value=_fake_response())
		return GoogleGateway(logger=logging.getLogger(), _client=client)

	@pytest.mark.asyncio
	async def test_success_extracts_text_and_usage(self, gateway):
		result = await gateway.generate(
			model=LLMModelType.gemini_flash_preview,
			system_prompt="sys",
			user_message="hi",
			history=[],
		)
		assert result.text == "gemini says hi"
		assert result.provider == "google"
		assert result.usage == {"prompt_token_count": 3, "candidates_token_count": 4}

		kwargs = gateway._client.aio.models.generate_content.call_args.kwargs
		assert kwargs["model"] == "gemini-3-flash-preview"
		# history empty -> contents is just the user message
		assert kwargs["contents"][-1] == "hi"

	@pytest.mark.asyncio
	async def test_history_mapped_to_contents(self, gateway):
		history = [
			UserMessageDTO(
				chat_id=uuid4(), message="u1", llm_model=LLMModelType.gemini_flash_preview, role=ChatRoles.USER
			),
			UserMessageDTO(
				chat_id=uuid4(), message="m1", llm_model=LLMModelType.gemini_flash_preview, role=ChatRoles.MODEL
			),
		]
		await gateway.generate(
			model=LLMModelType.gemini_flash_preview,
			system_prompt="sys",
			user_message="u2",
			history=history,
		)
		contents = gateway._client.aio.models.generate_content.call_args.kwargs["contents"]
		# two Content objects from history + the final user string
		assert len(contents) == 3
		assert contents[-1] == "u2"

	@pytest.mark.asyncio
	async def test_auth_error_maps_to_authentication_exception(self, gateway):
		gateway._client.aio.models.generate_content.side_effect = Exception("401 Permission denied")
		with pytest.raises(AuthenticationException):
			await gateway.generate(
				model=LLMModelType.gemini_flash_preview,
				system_prompt="sys",
				user_message="hi",
				history=[],
			)
