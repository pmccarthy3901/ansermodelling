import numpy as np 
import torch 
from torch.utils.data import Dataset, DataLoader

class AnserDataset(Dataset):
    '''
        Subclasses torch dataset to make the data usable by torch.

        Parameters
        ---------- 
        path : string
            path to dataset npz file.
    '''

    def __init__(self,path, x_mean = None, x_std = None):
        data = np.load(path)
        
        self.x = torch.tensor(data["xs"], dtype=torch.float32)
        self.y = torch.tensor(data["ys"], dtype=torch.float32)

        self.x_mean = x_mean
        self.x_std = x_std

    def __len__(self):
        return len(self.x)

    def __getitem__(self,idx):

        x = self.x[idx]
        
        if self.x_mean is not None and self.x_std is not None:
            x = (x - self.x_mean) / self.x_std

        return x, self.y[idx]




def make_dataloaders(path, train_frac = 0.8, batch_size = 32):

    full_dataset = AnserDataset(path)

    n = len(full_dataset)
    n_train = int(train_frac*n)
    n_test = n - n_train 

    indices = torch.randperm(n)

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    

    x_mean = full_dataset.x[train_idx].mean(dim = 0)
    x_std = full_dataset.x[train_idx].std(dim = 0)

    train_ds = AnserDataset(path,x_mean = x_mean, x_std = x_std)
    train_ds.x = train_ds.x[train_idx]
    train_ds.y = train_ds.y[train_idx]

    test_ds = AnserDataset(path,x_mean = x_mean, x_std = x_std)
    test_ds.x = test_ds.x[test_idx]
    test_ds.y = test_ds.y[test_idx]

    train_dataloader = DataLoader(train_ds, batch_size = batch_size)
    test_dataloader = DataLoader(test_ds,batch_size = batch_size)

    return train_dataloader, test_dataloader
