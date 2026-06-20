import logging

from dishka import Provider, Scope, make_async_container, provide

from src.application.agent.service import AgentService
from src.application.anthropic.service import AnthropicService
from src.application.deepseek.service import DeepSeekService
from src.application.google.service import GoogleService
from src.application.mock.service import MockService
from src.application.ports import IAgentService
from src.application.zai.service import ZaiService
from src.domain.models import LLMProvider
from src.infrastructure.exception_handler import ExceptionHandler
from src.infrastructure.gateways.anthropic_gateway import AnthropicGateway
from src.infrastructure.gateways.deepseek_gateway import DeepSeekGateway
from src.infrastructure.gateways.google_gateway import GoogleGateway
from src.infrastructure.gateways.mock_gateway import MockGateway
from src.infrastructure.gateways.zai_gateway import ZaiGateway

logger = logging.getLogger(__name__)


# Gateways are APP-scoped singletons: the service is stateless (no DB session),
# and each gateway holds a long-lived, concurrency-safe SDK client built once
# from settings in __post_init__. No shared AsyncOpenAI type is injected, so
# there is no dishka key ambiguity between the Z.ai and DeepSeek gateways.


class GatewayProvider(Provider):
	@provide(scope=Scope.APP)
	def anthropic_gateway(self) -> AnthropicGateway:
		return AnthropicGateway(logger=logger)

	@provide(scope=Scope.APP)
	def google_gateway(self) -> GoogleGateway:
		return GoogleGateway(logger=logger)

	@provide(scope=Scope.APP)
	def zai_gateway(self) -> ZaiGateway:
		return ZaiGateway(logger=logger)

	@provide(scope=Scope.APP)
	def deepseek_gateway(self) -> DeepSeekGateway:
		return DeepSeekGateway(logger=logger)

	@provide(scope=Scope.APP)
	def mock_gateway(self) -> MockGateway:
		return MockGateway(logger=logger)


class ServiceProvider(Provider):
	@provide(scope=Scope.APP)
	def anthropic_service(self, gateway: AnthropicGateway) -> AnthropicService:
		return AnthropicService(_gateway=gateway)

	@provide(scope=Scope.APP)
	def google_service(self, gateway: GoogleGateway) -> GoogleService:
		return GoogleService(_gateway=gateway)

	@provide(scope=Scope.APP)
	def zai_service(self, gateway: ZaiGateway) -> ZaiService:
		return ZaiService(_gateway=gateway)

	@provide(scope=Scope.APP)
	def deepseek_service(self, gateway: DeepSeekGateway) -> DeepSeekService:
		return DeepSeekService(_gateway=gateway)

	@provide(scope=Scope.APP)
	def mock_service(self, gateway: MockGateway) -> MockService:
		return MockService(_gateway=gateway)


class InfrastructureProvider(Provider):
	@provide(scope=Scope.APP)
	def exception_handler(self) -> ExceptionHandler:
		return ExceptionHandler(logger=logger)


class AgentServiceProvider(Provider):
	@provide(scope=Scope.APP)
	def agent_service(
		self,
		anthropic: AnthropicService,
		google: GoogleService,
		zai: ZaiService,
		deepseek: DeepSeekService,
		mock: MockService,
	) -> IAgentService:
		return AgentService(
			provider_services={
				LLMProvider.ANTHROPIC: anthropic,
				LLMProvider.GOOGLE: google,
				LLMProvider.ZAI: zai,
				LLMProvider.DEEPSEEK: deepseek,
				LLMProvider.MOCK: mock,
			}
		)


def create_container():
	"""Create the dishka container wiring gateways, provider services, and the dispatcher."""
	return make_async_container(
		GatewayProvider(),
		ServiceProvider(),
		InfrastructureProvider(),
		AgentServiceProvider(),
	)
