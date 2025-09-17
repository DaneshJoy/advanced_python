import random
import asyncio


async def eeg_producer(queues, n_channel=4, sampling_rate=10, duration=1):
    n_samples = sampling_rate * duration
    for i in range(n_samples):
        sample = [random.gauss(0, 1) for _ in range(n_channel)]
        print(f"Produced {i}: {sample}")
        for q in queues:
            await q.put((i, sample))
            await asyncio.sleep(5*(1/sampling_rate))


async def eeg_consumer(queue, n_samples=4, consumer_id=0):
    for _ in range(n_samples):
        i, sample = await queue.get()
        print(f"Consumer {consumer_id} consumed {i}: {sample}")
        queue.task_done()


async def run_async_pubsub(n_channel=4, sampling_rate=10,
                           duration=1, n_consumers=2):
    n_samples = sampling_rate * duration
    queues = [asyncio.Queue() for _ in range(n_consumers)]
    producer_task = asyncio.create_task(eeg_producer(queues, n_channel,
                                                     sampling_rate, duration))
    consumers = [
        asyncio.create_task(eeg_consumer(q, n_samples=n_samples,
                                         consumer_id=i))
        for i, q in enumerate(queues)
    ]
    await asyncio.gather(*consumers)
    for q in queues:
        await q.join()
    for c in consumers:
        c.cancel()
    producer_task.cancel()


if __name__ == "__main__":
    asyncio.run(run_async_pubsub(duration=1, n_consumers=2))
