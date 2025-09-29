import logging

logging.basicConfig(
    filename="eeg.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)

def generate_eeg(n=3):
    logger.info("start generator")
    for i in range(n):
        logger.info("sample %d", i)
        yield f"sample_{i}"

# run — logs go to eeg.log
for _ in generate_eeg(2):
    pass