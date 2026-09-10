import torch
from torch.utils.data import Dataset, DataLoader

class HousePricesDataset(Dataset):
    def __init__(self, X, y = None):
        

        if hasattr(X, 'toarray'):
            X = X.toarray()
        self.X = torch.tensor(X, dtype= torch.float32)

        if y is not None:
            self.y = torch.tensor(y, dtype= torch.float32)



    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):

        if self.y is not None:
            return self.X[index], self.y[index]
        return self.X[index]


def create_dataloader(dataset, batch_size, shuffle = True):
    return DataLoader(dataset= dataset, batch_size= batch_size, shuffle= shuffle)