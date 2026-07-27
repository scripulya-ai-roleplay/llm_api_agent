import asyncio
import json
import logging

import redis.asyncio

logger = logging.getLogger(__name__)


class TokenStream:
	"""Best-effort firehose of generation deltas to a Redis Pub/Sub channel.

	Carries two delta kinds over one queue: "token" (the answer) and "thinking"
	(the model's chain-of-thought). A single monotonic seq numbers every frame so
	the downstream relay can interleave them in emission order if it needs to.
	"""

	def __init__(self, redis_client: redis.asyncio.Redis, channel: str) -> None:
		self._redis = redis_client
		self._channel = channel
		self._queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()
		self._task: asyncio.Task[None] | None = None
		self._seq = 0

	async def __aenter__(self) -> "TokenStream":
		if self._channel:
			self._task = asyncio.create_task(self._drain())
		return self

	async def emit(self, text: str) -> None:
		if self._task is None or not text:
			return
		self._queue.put_nowait(("token", text))

	async def emit_thinking(self, text: str) -> None:
		if self._task is None or not text:
			return
		self._queue.put_nowait(("thinking", text))

	async def _drain(self) -> None:
		try:
			while True:
				item = await self._queue.get()
				if item is None:  # sentinel: flush complete
					return
				kind, text = item
				self._seq += 1
				try:
					await self._redis.publish(
						self._channel,
						json.dumps({"type": kind, "seq": self._seq, "text": text}),
					)
				except Exception:
					logger.debug("delta publish failed; dropped seq=%s", self._seq, exc_info=True)
		except Exception:
			logger.debug("delta drain loop failed", exc_info=True)

	async def __aexit__(self, exc_type, exc, tb) -> None:
		if self._task is None:
			return
		self._queue.put_nowait(None)
		try:
			await self._task
		except Exception:
			logger.debug("delta drain task raised on exit", exc_info=True)
