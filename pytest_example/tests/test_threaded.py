import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from eeg.threaded import threaded_producer, threaded_consumer


def test_threaded_producer_consumer():
    q = queue.Queue()
    n = 4

    producer_thread = threading.Thread(target=threaded_producer, args=(q, n))
    producer_thread.start()
    producer_thread.join()
    
    result = threaded_consumer(q, n)
    
    assert q.get() == None
    assert result == list(range(n))

def test_empty_queue():
    q = queue.Queue()
    n = 0

    producer_thread = threading.Thread(target=threaded_producer, args=(q, n))
    producer_thread.start()
    producer_thread.join()
    
    result = threaded_consumer(q, n)

    assert result == []

def test_consumer_results_1():
    q = queue.Queue()
    n = 4

    producer_thread = threading.Thread(target=threaded_producer, args=(q, n))
    producer_thread.start()
    producer_thread.join()
    
    results = []
    def consumer_wrapper():
        results.extend(threaded_consumer(q, n))

    consumer_thread = threading.Thread(target=consumer_wrapper)
    consumer_thread.start()
    consumer_thread.join()

    assert results == list(range(n))


def test_consumer_results_2():
    q = queue.Queue()
    n = 4
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        _ = executor.submit(threaded_producer, q, n)
        future_consumer = executor.submit(threaded_consumer, q, n)
        result = future_consumer.result()

    assert result == list(range(n))