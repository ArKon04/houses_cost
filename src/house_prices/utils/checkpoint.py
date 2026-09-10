from pathlib import Path
import torch

class CheckpointManager:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = float("inf")

    def save_best(self, model, optimizer, epoch, metric):
        if metric < self.best_metric:
            self.best_metric = metric

            checkpoint = {
                "epoch": epoch,
                "model_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metric": metric
            }

            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
            return True
        return False

    def load_best(self, model, optimizer = None):
        checkpoint = torch.load(self.checkpoint_dir / "best.pt")
        model.load_state_dict(checkpoint['model_dict'])

        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']