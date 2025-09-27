import asyncio, random


async def async_producer(n):
    await asyncio.sleep(random.random())
    return [i for i in range(n)]