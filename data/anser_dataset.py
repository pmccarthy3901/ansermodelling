import numpy as np 
import torch 
from torch.utils.data import Dataset 


class AnserDataset(Dataset):
    '''
        Subclasses torch dataset to make the data usable by torch.

        Parameters
        ---------- 
        path : string
            path to dataset npz file.
    '''

    def __init__(self,path):
        data = np.load(path)
        self.x = torch.tensor(data["xs"], dtype=torch.float32)
        self.y = torch.tensor(data["ys"], dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]

