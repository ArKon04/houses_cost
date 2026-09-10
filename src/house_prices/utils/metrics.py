import torch


def rmse(predictions, targets):
    mse = torch.mean((predictions - targets)**2)
    return torch.sqrt(mse) 