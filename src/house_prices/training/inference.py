import numpy as np
import torch

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    predictions = []

    for X in loader:
        X = X.to(device)
        prediction = model(X)

        predictions.append(prediction.squeeze(1).cpu())
    predictions = torch.cat(predictions)
    return predictions.numpy()