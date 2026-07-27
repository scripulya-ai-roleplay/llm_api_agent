import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


async def emit_token(on_token: Callable[[str], Awaitable[None]] | None, text: str) -> None:
	await _emit(on_token, text)


async def emit_thinking(on_thinking: Callable[[str], Awaitable[None]] | None, text: str) -> None:
	await _emit(on_thinking, text)


async def _emit(sink: Callable[[str], Awaitable[None]] | None, text: str) -> None:
	"""Forward a generated delta (answer text or reasoning) to a streaming sink,
	swallowing errors.

	Streaming is decorative: a Redis blip in the backend's relay must never abort
	the in-flight generation. Mirrors the heartbeat's own best-effort philosophy.
	"""
	if sink is None or not text:
		return
	try:
		await sink(text)
	except Exception:
		logger.debug("streaming sink failed; delta dropped", exc_info=True)
