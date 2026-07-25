import asyncio
import logging

import redis.asyncio

from src.conf import settings

logger = logging.getLogger(__name__)

_DONE_TTL_SECONDS = 3600


def _alive_key(request_id: str) -> str:
	return f"gen:{request_id}:alive"


def _done_key(request_id: str) -> str:
	return f"gen:{request_id}:done"


def tokens_key(request_id: str) -> str:
	return f"gen:{request_id}:tokens"


class Heartbeat:
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
		try:
			await self._redis.set(_alive_key(self._request_id), "1", ex=settings.LLM_HEARTBEAT_ALIVE_TTL)
			while True:
				await asyncio.sleep(settings.LLM_HEARTBEAT_REFRESH_INTERVAL_SECONDS)
				await self._redis.set(_alive_key(self._request_id), "1", ex=settings.LLM_HEARTBEAT_ALIVE_TTL)
		except Exception:
			logger.warning("heartbeat refresh failed rid=%s", self._request_id, exc_info=True)


async def mark_done(redis_client: redis.asyncio.Redis, request_id: str) -> None:
	if not request_id:
		return
	try:
		pipe = redis_client.pipeline(transaction=True)
		pipe.set(_done_key(request_id), "done", ex=_DONE_TTL_SECONDS)
		pipe.delete(_alive_key(request_id))
		await pipe.execute()
	except Exception:
		logger.warning("heartbeat mark_done failed rid=%s", request_id, exc_info=True)
