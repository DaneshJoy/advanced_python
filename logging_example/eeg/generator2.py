import functools
import time
import logging

logger = logging.getLogger(__name__)

def log_calls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info("ENTER %s args=%s kwargs=%s", func.__name__, args, kwargs)
        start = time.time()
        try:
            for item in func(*args, **kwargs):     # assumes func is a generator
                logger.debug("YIELD %r", item)
                yield item
            logger.info("EXIT %s (%.3fs)", func.__name__, time.time() - start)
        except Exception:
            logger.exception("EXCEPTION in %s", func.__name__)
            raise
    return wrapper

@log_calls
def generate_eeg(n=3):
    for i in range(n):
        yield f"sample_{i}"

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    for s in generate_eeg(2):
        print(s)