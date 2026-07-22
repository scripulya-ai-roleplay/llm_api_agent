import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.conf import settings
from src.infrastructure.heartbeat import Heartbeat, mark_done


@pytest.mark.unit
class TestHeartbeat:
	@pytest.mark.asyncio
	async def test_sets_alive_and_stops_on_exit(self):
		redis_mock = AsyncMock()
		hb = Heartbeat(redis_mock, "rid-1")

		async with hb:
			await asyncio.sleep(0)  # let the refresh task run its first SET

		redis_mock.set.assert_awaited()
		args, kwargs = redis_mock.set.await_args
		assert args[0] == "gen:rid-1:alive"
		assert kwargs["ex"] == settings.LLM_HEARTBEAT_ALIVE_TTL
		assert hb._task is not None and hb._task.cancelled()  # refresh stopped on exit

	@pytest.mark.asyncio
	async def test_swallows_redis_errors(self, monkeypatch):
		monkeypatch.setattr(settings, "LLM_HEARTBEAT_REFRESH_INTERVAL_SECONDS", 0.01)
		redis_mock = AsyncMock()
		redis_mock.set.side_effect = RuntimeError("redis down")

		async with Heartbeat(redis_mock, "rid-1"):  # must not raise
			await asyncio.sleep(0.02)

	@pytest.mark.asyncio
	async def test_noop_without_request_id(self):
		redis_mock = AsyncMock()
		hb = Heartbeat(redis_mock, "")

		async with hb:
			await asyncio.sleep(0)

		redis_mock.set.assert_not_awaited()
		assert hb._task is None


@pytest.mark.unit
class TestMarkDone:
	@pytest.mark.asyncio
	async def test_sets_done_and_deletes_alive(self):
		pipe = MagicMock()
		pipe.execute = AsyncMock()
		redis_mock = MagicMock()
		redis_mock.pipeline.return_value = pipe

		await mark_done(redis_mock, "rid-1")

		pipe.set.assert_called_once_with("gen:rid-1:done", "done", ex=3600)
		pipe.delete.assert_called_once_with("gen:rid-1:alive")
		pipe.execute.assert_awaited_once()

	@pytest.mark.asyncio
	async def test_noop_without_request_id(self):
		redis_mock = MagicMock()
		await mark_done(redis_mock, "")
		redis_mock.pipeline.assert_not_called()

	@pytest.mark.asyncio
	async def test_swallows_redis_errors(self):
		redis_mock = MagicMock()
		redis_mock.pipeline.side_effect = RuntimeError("redis down")
		await mark_done(redis_mock, "rid-1")  # must not raise
