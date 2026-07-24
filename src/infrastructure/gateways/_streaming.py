import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


async def emit_token(on_token: Callable[[str], Awaitable[None]] | None, text: str) -> None:
	"""Forward a generated text delta to the streaming sink, swallowing errors.

	Streaming is decorative: a Redis blip in the backend's relay must never abort
	the in-flight generation. Mirrors the heartbeat's own best-effort philosophy.
	"""
	if on_token is None or not text:
		return
	try:
		await on_token(text)
	except Exception:
		logger.debug("on_token sink failed; token dropped", exc_info=True)
