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
        x_mean : float 
            mean x value used for input normalisation.
        x_std : float 
            stdev of x used for input normalisation.
        noise_std : float 
            standard deviation of noise to be added to inputs. Default = 0
    '''

    def __init__(self,
                 path, 
                 x_mean : float = None, 
                 x_std : float = None, 
                 noise_std : float = 0):
        data = np.load(path)
        
        self.x = torch.tensor(data["xs"], dtype=torch.float32)
        self.y = torch.tensor(data["ys"], dtype=torch.float32)

        self.x_mean = x_mean
        self.x_std = x_std
        self.noise_std = noise_std

    def __len__(self):
        return len(self.x)

    def __getitem__(self,idx):

        x = self.x[idx]
        y = self.y[idx]

        if self.x_mean is not None and self.x_std is not None:
            x = (x - self.x_mean) / self.x_std
        
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        return x, y




def make_dataloaders(path : str,
                     train_frac : float = 0.8,
                     batch_size : int = 32,
                     normalise_data : bool = False,
                     noise_std : float = 0):

    '''
        Makes training and testing dataloaders from dataset npz file.

        Parameters 
        ---------- 
        path : string 
            path to dataset npz file 
        train_frac : float 
            fraction of data used for training, default = 0.8
        batch_size : int 
            Batch size, default = 32

        normalise_data : bool
            Choose whether to normalise test data, default = False
        noise_std : float 
            standard deviation of noise to be added to inputs. Default = 0
    
        Returns 
        ------- 
        train_dataloader, test_dataloder 
            Pytorch dataloaders generated from the data
    '''
    full_dataset = AnserDataset(path)

    n = len(full_dataset)
    n_train = int(train_frac*n)
    n_test = n - n_train 

    indices = torch.randperm(n)

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    if normalise_data:
        x_mean = full_dataset.x[train_idx].mean(dim = 0)
        x_std = full_dataset.x[train_idx].std(dim = 0)

        train_ds = AnserDataset(path,x_mean = x_mean, x_std = x_std, noise_std = noise_std)
        train_ds.x = train_ds.x[train_idx]
        train_ds.y = train_ds.y[train_idx]

        test_ds = AnserDataset(path,x_mean = x_mean, x_std = x_std)
        test_ds.x = test_ds.x[test_idx]
        test_ds.y = test_ds.y[test_idx]
    else:
        train_ds = AnserDataset(path,noise_std = noise_std)
        train_ds.x = train_ds.x[train_idx]
        train_ds.y = train_ds.y[train_idx]

        test_ds = AnserDataset(path)
        test_ds.x = test_ds.x[test_idx]
        test_ds.y = test_ds.y[test_idx]


    train_dataloader = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(test_ds,batch_size = batch_size)

    return train_dataloader, test_dataloader
