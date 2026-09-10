from pathlib import Path
from datetime import datetime



def create_experiment_dir(base_dir, experiment_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(base_dir) / f"{timestamp}_{experiment_name}"

    path.mkdir(parents=True, exist_ok=True)
    return path