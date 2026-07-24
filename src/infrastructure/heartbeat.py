import asyncio
import logging

import redis.asyncio

from src.conf import settings

logger = logging.getLogger(__name__)

# Lingers ~1h so a late backend sweep or a redelivery can't re-FAIL a request the
# agent already finished. Mirrors the backend watchdog's own marker.
_DONE_TTL_SECONDS = 3600


def _alive_key(request_id: str) -> str:
	return f"gen:{request_id}:alive"


def _done_key(request_id: str) -> str:
	return f"gen:{request_id}:done"


def tokens_key(request_id: str) -> str:
	# Shared CONTRACT with scripulya_ai (it subscribes here to relay tokens to SSE).
	# The agent publishes {"type":"token"|"done"|"error", ...} frames as JSON strings.
	return f"gen:{request_id}:tokens"


class Heartbeat:
	"""Best-effort: Redis errors are swallowed so an outage never breaks generation.
	A falsy request_id (the backend omitted a correlation_id) disables the heartbeat."""

	def __init__(self, redis_client: redis.asyncio.Redis, request_id: str):
		self._redis = redis_client
		self._request_id = request_id
		self._task: asyncio.Task[None] | None = None

	async def __aenter__(self) -> "Heartbeat":
		if self._request_id:
			self._task = asyncio.create_task(self._refresh_loop())
		return self

	async def __aexit__(self, exc_type, exc, tb) -> None:
		if self._task is None:
			return
		self._task.cancel()
		try:
			await self._task
		except asyncio.CancelledError:
			pass

	async def _refresh_loop(self) -> None:
		# Set before the first sleep so the key exists immediately on enter.
		try:
			await self._redis.set(_alive_key(self._request_id), "1", ex=settings.LLM_HEARTBEAT_ALIVE_TTL)
			while True:
				await asyncio.sleep(settings.LLM_HEARTBEAT_REFRESH_INTERVAL_SECONDS)
				await self._redis.set(_alive_key(self._request_id), "1", ex=settings.LLM_HEARTBEAT_ALIVE_TTL)
		except Exception:
			logger.warning("heartbeat refresh failed rid=%s", self._request_id, exc_info=True)


async def mark_done(redis_client: redis.asyncio.Redis, request_id: str) -> None:
	# The done marker is shared with the backend watchdog, so a finished request is
	# never re-FAILed. No-op without a request_id.
	if not request_id:
		return
	try:
		pipe = redis_client.pipeline(transaction=True)
		pipe.set(_done_key(request_id), "done", ex=_DONE_TTL_SECONDS)
		pipe.delete(_alive_key(request_id))
		await pipe.execute()
	except Exception:
		logger.warning("heartbeat mark_done failed rid=%s", request_id, exc_info=True)
