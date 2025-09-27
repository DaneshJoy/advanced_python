import queue


def threaded_producer(q: queue.Queue, n: int):
    for i in range(n):
        q.put(i)
    q.put(None)  # Sentinel value to indicate completion
    
def threaded_consumer(q: queue.Queue, n: int):
    return [q.get() for _ in range(n)]
