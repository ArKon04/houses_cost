import logging
from pathlib import Path

def setup_logger(log_dir):
    Path(log_dir).mkdir(
        parents=True,
        exist_ok=True
    )

    logger = logging.getLogger('house_prices')
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)


    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(f"{log_dir}/train.log")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger