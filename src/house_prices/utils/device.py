import torch

def get_device(device_name):
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    return torch.device(device_name)