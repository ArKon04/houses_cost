import pandas as pd
from pathlib import Path


def load_train_test(config):
    train_path = Path(config.paths.train)
    test_path = Path(config.paths.test)

    if not train_path.exists():
        raise FileNotFoundError(f"Train not found: {train_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"Test not found {test_path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)


    return train, test


def load_test(config):

    test_path = Path(config.paths.test)

    if not test_path.exists():
        raise FileNotFoundError(f"test file not found: {test_path}")

    return pd.read_csv(test_path)