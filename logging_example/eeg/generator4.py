import logging
from logging import StreamHandler, FileHandler

def setup_logging(logfile="eeg.log", level=logging.DEBUG):
    root = logging.getLogger()
    root.setLevel(level)
    if root.handlers:   # avoid duplicate handlers on re-import
        return root
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s",
                            "%Y-%m-%d %H:%M:%S")

    fh = FileHandler(logfile)
    fh.setLevel(level)
    fh.setFormatter(fmt)

    ch = StreamHandler()           # console
    ch.setLevel(logging.INFO)      # console less noisy than file
    ch.setFormatter(fmt)

    root.addHandler(fh)
    root.addHandler(ch)
    return root

# use decorator from step 2 (assumes it's available)
if __name__ == "__main__":
    setup_logging()   # must call early
    logger = logging.getLogger(__name__)

    def generate_eeg(n=4):
        for i in range(n):
            logger.debug("creating sample %d", i)
            logger.info("Sample info message")
            yield f"sample_{i}"

    for s in generate_eeg(3):
        print("got:", s)