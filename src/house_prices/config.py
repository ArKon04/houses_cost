from pathlib import Path
from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str | Path) -> DictConfig:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file was note found: {config_path}"
        )

    config = OmegaConf.load(config_path)
    OmegaConf.resolve(config)

    return config


def save_config(config, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, path)
