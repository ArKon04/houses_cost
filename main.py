from house_prices.config import load_config, save_config
from house_prices.data.pipeline import  prepare_dataloaders
from house_prices.training.trainer import Trainer
from house_prices.utils.seed import seed_everything
from house_prices.utils.device import get_device
from house_prices.utils.logger import setup_logger
from house_prices.utils.checkpoint import CheckpointManager

from house_prices.models.mlp import MLP

import torch.optim as optim
import torch


def main() -> None:
    config = load_config("config/config.yaml")
    seed_everything(config.general.seed)
    device = get_device(config.general.device)

    logger = setup_logger(config.paths.output_dir)
    checkpoint_manager = CheckpointManager(config.paths.checkpoint)

    save_config(config, f"{config.paths.output_dir}/config.yaml")

    train_loader, val_loader = prepare_dataloaders(config)

    batch_x, batch_y = next(iter(train_loader))
    model = MLP(input_dim = batch_x.shape[1], hidden_dims = config.model.hidden_dims)
    


    optimizer = optim.Adam(model.parameters(), lr = config.training.learning_rate)
    loss = torch.nn.MSELoss()
    trainer = Trainer(model, optimizer, loss, device = device, checkpoint_path= config.paths.checkpoint, logger = logger, checkpoint_manager = checkpoint_manager)

    trainer.fit(train_loader, val_loader, epochs = config.training.epochs)

    

    

    

    

    




if __name__ == "__main__":
    main()


