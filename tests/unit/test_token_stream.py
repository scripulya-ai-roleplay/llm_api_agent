import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from src.infrastructure.token_stream import TokenStream


@pytest.mark.unit
class TestTokenStream:
	@pytest.mark.asyncio
	async def test_publishes_tokens_in_order_with_seq(self):
		published: list[dict] = []

		class _Redis:
			async def publish(self, channel, payload):
				published.append(json.loads(payload))

		async with TokenStream(_Redis(), "gen:rid:tokens") as ts:
			await ts.emit("a")
			await ts.emit("bb")
			await ts.emit("ccc")

		assert [p["text"] for p in published] == ["a", "bb", "ccc"]
		assert [p["seq"] for p in published] == [1, 2, 3]
		assert all(p["type"] == "token" for p in published)

	@pytest.mark.asyncio
	async def test_emit_thinking_publishes_thinking_frames_interleaved(self):
		published: list[dict] = []

		class _Redis:
			async def publish(self, channel, payload):
				published.append(json.loads(payload))

		async with TokenStream(_Redis(), "gen:rid:tokens") as ts:
			await ts.emit_thinking("hmm")
			await ts.emit("answer")
			await ts.emit_thinking("...")

		# Kinds are preserved and a single monotonic seq orders all frames, token
		# and thinking alike, so the relay can interleave them in emission order.
		assert [(p["type"], p["text"]) for p in published] == [
			("thinking", "hmm"),
			("token", "answer"),
			("thinking", "..."),
		]
		assert [p["seq"] for p in published] == [1, 2, 3]

	@pytest.mark.asyncio
	async def test_flushes_every_token_before_exit(self):
		# The drain publishes slower than tokens are enqueued; __aexit__ must wait
		# for it to catch up so no token is left unpublished (and the caller's
		# terminal frame lands after the tokens).
		published: list[dict] = []

		class _Redis:
			async def publish(self, channel, payload):
				await asyncio.sleep(0.001)
				published.append(json.loads(payload))

		async with TokenStream(_Redis(), "gen:rid:tokens") as ts:
			for i in range(25):
				await ts.emit(str(i))

		assert [p["text"] for p in published] == [str(i) for i in range(25)]

	@pytest.mark.asyncio
	async def test_swallows_redis_errors(self):
		redis_mock = AsyncMock()
		redis_mock.publish.side_effect = RuntimeError("redis down")

		async with TokenStream(redis_mock, "gen:rid:tokens") as ts:  # must not raise
			await ts.emit("a")
			await ts.emit("b")

		# Both publishes were attempted (and failed) despite the errors.
		assert redis_mock.publish.await_count == 2

	@pytest.mark.asyncio
	async def test_noop_without_channel(self):
		redis_mock = AsyncMock()
		async with TokenStream(redis_mock, "") as ts:
			await ts.emit("a")
		redis_mock.publish.assert_not_awaited()
		assert ts._task is None
