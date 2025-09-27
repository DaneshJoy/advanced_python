import pytest
from eeg.async_producer import async_producer


@pytest.mark.asyncio
async def test_async_producer():
    result = await async_producer(4)
    assert result == [0, 1, 2, 3]
    assert len(result) == 4
