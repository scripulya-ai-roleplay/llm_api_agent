import logging
import uuid
from contextvars import ContextVar

# Correlation id for the current message; populated from the incoming
# RabbitMQ message (or a generated fallback) so log lines can be traced.
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
	return correlation_id_var.get()


def set_correlation_id(correlation_id: str) -> None:
	correlation_id_var.set(correlation_id)


def ensure_correlation_id() -> str:
	"""Return the current correlation id, generating one if none was set
	so traces are never orphaned.
	"""

	value = correlation_id_var.get()
	if not value:
		value = str(uuid.uuid4())
		correlation_id_var.set(value)
	return value


class CorrelationIdFilter(logging.Filter):
	"""Attach the current correlation_id to every log record as `correlation_id`."""

	def filter(self, record: logging.LogRecord) -> bool:
		record.correlation_id = correlation_id_var.get()  # type: ignore[attr-defined]
		return True
