import asyncio

import pytest

from worker.pipeline.model_manager import ModelManager
from worker.presets.base import Mode


@pytest.mark.asyncio
async def test_initial_mode_is_none():
    mm = ModelManager()
    assert mm.current_mode() is None
    assert mm.swap_count == 0


@pytest.mark.asyncio
async def test_ensure_first_call_sets_mode():
    mm = ModelManager()
    await mm.ensure(Mode.IMAGE)
    assert mm.current_mode() is Mode.IMAGE
    assert mm.swap_count == 0  # first load is not a swap


@pytest.mark.asyncio
async def test_ensure_same_mode_is_noop():
    mm = ModelManager()
    await mm.ensure(Mode.IMAGE)
    await mm.ensure(Mode.IMAGE)
    assert mm.swap_count == 0


@pytest.mark.asyncio
async def test_ensure_switch_increments_swap():
    mm = ModelManager()
    await mm.ensure(Mode.IMAGE)
    await mm.ensure(Mode.VIDEO)
    assert mm.current_mode() is Mode.VIDEO
    assert mm.swap_count == 1


@pytest.mark.asyncio
async def test_ensure_is_serialized():
    mm = ModelManager()

    async def call(m):
        await mm.ensure(m)

    await asyncio.gather(call(Mode.IMAGE), call(Mode.VIDEO), call(Mode.IMAGE))
    assert mm.current_mode() in (Mode.IMAGE, Mode.VIDEO)
