import asyncio
import logging

from src.app import app
from src.conf import settings
from src.infrastructure.logging.trace import CorrelationIdFilter

logger = logging.getLogger(__name__)


def setup_logging() -> None:
	level = logging.DEBUG if settings.DEBUG else logging.INFO
	logging.basicConfig(
		level=level,
		format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
	)
	# Attach the correlation-id filter to existing handlers so every record
	# carries `correlation_id` (empty until set per message).
	for handler in logging.getLogger().handlers:
		handler.addFilter(CorrelationIdFilter())


def main() -> int:
	setup_logging()
	logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)
	try:
		asyncio.run(app.run())
		return 0
	except KeyboardInterrupt:
		logger.info("Interrupted, shutting down")
		return 0
	except Exception:
		logger.exception("Failed to start service")
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
