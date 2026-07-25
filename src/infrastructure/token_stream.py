import asyncio
import json
import logging

import redis.asyncio

logger = logging.getLogger(__name__)


class TokenStream:
	def __init__(self, redis_client: redis.asyncio.Redis, channel: str) -> None:
		self._redis = redis_client
		self._channel = channel
		self._queue: asyncio.Queue[tuple[int, str] | None] = asyncio.Queue()
		self._task: asyncio.Task[None] | None = None
		self._seq = 0

	async def __aenter__(self) -> "TokenStream":
		if self._channel:
			self._task = asyncio.create_task(self._drain())
		return self

	async def emit(self, text: str) -> None:
		if self._task is None or not text:
			return
		self._seq += 1
		self._queue.put_nowait((self._seq, text))

	async def _drain(self) -> None:
		try:
			while True:
				item = await self._queue.get()
				if item is None:  # sentinel: flush complete
					return
				seq, text = item
				try:
					await self._redis.publish(
						self._channel,
						json.dumps({"type": "token", "seq": seq, "text": text}),
					)
				except Exception:
					logger.debug("token publish failed; token dropped seq=%s", seq, exc_info=True)
		except Exception:
			logger.debug("token drain loop failed", exc_info=True)

	async def __aexit__(self, exc_type, exc, tb) -> None:
		if self._task is None:
			return
		self._queue.put_nowait(None)
		try:
			await self._task
		except Exception:
			logger.debug("token drain task raised on exit", exc_info=True)
