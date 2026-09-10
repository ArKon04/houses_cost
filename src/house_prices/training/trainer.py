import torch
from house_prices.utils.metrics import rmse

class Trainer:
    def __init__(self, model, optimizer, loss, device, checkpoint_path, logger, checkpoint_manager):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss = loss
        self.device = device
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager



    def train_epoch(self, loader):
        self.model.train()

        total_loss = 0
        for X, y in loader:

            X = X.to(self.device)
            y = y.to(self.device)

            prediction = self.model(X)

            loss = self.loss(prediction.squeeze(), y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    

    def validate_epoch(self, loader):
        self.model.eval()

        predictions = []
        targets = []

        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                y = y.to(self.device)
                
                prediction = self.model(X).squeeze()
                predictions.append(prediction.cpu())
                targets.append(y.cpu())
                
        predictions = torch.cat(predictions)
        targets = torch.cat(targets)

        val_rmse = rmse(predictions, targets)
        return val_rmse


    def fit(self, train_loader, val_loader, epochs):
            for epoch in range(epochs):
                train_loss = self.train_epoch(train_loader)
                val_rmse = self.validate_epoch(val_loader)
                self.logger.info(f"""
                    Epoch {epoch}/{epochs}
                    Train loss: {train_loss:.4f}
                    Val rmse: {val_rmse:.4f}
                    """
                )

                saved = self.checkpoint_manager.save_best(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_rmse
                )
                if saved:
                    self.logger.info("new_checkpoint")

                
