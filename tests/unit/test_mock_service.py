import logging
from uuid import uuid4

import pytest

from src.application.mock.service import MockService
from src.application.ports import LLMRequest, UserMessageDTO
from src.domain.models import ChatRoles, LLMModelType
from src.infrastructure.gateways.mock_gateway import MockGateway


@pytest.mark.unit
class TestMockService:
	@pytest.fixture
	def service(self) -> MockService:
		# The mock gateway is fully offline, so use the real instance.
		return MockService(_gateway=MockGateway(logger=logging.getLogger()))

	@pytest.mark.asyncio
	async def test_returns_model_role_canned_reply(self, service):
		req = LLMRequest(
			message=UserMessageDTO(
				chat_id=uuid4(),
				message="hello world",
				llm_model=LLMModelType.testing_mock,
				role=ChatRoles.USER,
			)
		)
		result = await service.generate(req)

		assert result.role == ChatRoles.MODEL
		assert result.message == "Mock response for: hello world"
		assert result.chat_id == req.message.chat_id
		assert result.llm_model == LLMModelType.testing_mock
