import logging

from dishka.integrations.faststream import setup_dishka
from faststream import FastStream
from faststream.rabbit import RabbitBroker

from src.conf import settings
from src.controllers.llm import router as llm_router
from src.infrastructure.di import create_container

logger = logging.getLogger(__name__)


def create_broker() -> RabbitBroker:
	return RabbitBroker(settings.RABBIT_URL)


def create_app() -> FastStream:
	broker = create_broker()

	# Register the RPC-style subscriber + result publisher (controllers/llm.py).
	broker.include_router(llm_router)

	# NOTE: FastStream 0.6 removed direct AsyncAPI options from the constructor,
	# so we pass only the broker.
	app = FastStream(broker)

	# Wire dishka: auto_inject=True lets `FromDishka[...]` handler args resolve
	# without a per-handler @inject decorator. setup_dishka also enters/closes
	# the APP-scoped container across the app lifespan.
	container = create_container()
	setup_dishka(container=container, app=app, auto_inject=True)

	@app.on_startup
	async def on_startup() -> None:
		logger.info("agent up; in=%s out=%s", settings.LLM_REQUEST_QUEUE, settings.LLM_RESULT_QUEUE)

	@app.on_shutdown
	async def on_shutdown() -> None:
		logger.info("agent shutting down")
		await container.close()

	return app


app = create_app()
