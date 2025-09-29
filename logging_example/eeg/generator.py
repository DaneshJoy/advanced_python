import logging

logging.basicConfig(level=logging.INFO)  # call once at program start
logger = logging.getLogger(__name__)

def generate_eeg(n=5):
    for i in range(n):
        sample = f"eeg_sample_{i}"
        logger.info("generated sample %d", i)
        yield sample

if __name__ == "__main__":
    for s in generate_eeg(3):
        print('-----')