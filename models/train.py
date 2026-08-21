import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from data.reparametrisation import angles_to_normal_t
class PoseLoss(nn.Module):
    '''
    Loss function which gives a weighted sum between positional and angular loss.

    Parameters
    ---------- 
    w : float 
        Weight applied to the orientation component of the loss. Default = 0.1875/(pi^2)

    '''

    def __init__(self,
                 w: float = 0.1875/(np.pi**2)):
        super().__init__()
        self.w = w

    def forward(self,
                pred : torch.Tensor,
                true : torch.Tensor
                ) -> torch.Tensor: 
        '''
        Parameters
        ---------- 
        pred : torch.Tensor, shape(...,6)
            Predicted pose. 
        true : torch.Tensor, shape(...,6) 
            True Pose 

        Returns 
        ------- 
        loss : torch.Tensor 
            Weighted sum MSE loss. 
        '''
        
        x_pred = pred[...,:3]
        n_pred = pred[...,3:]

        x_true = true[...,:3]
        n_true = true[...,3:]

        #Positional loss 
        x_loss = ((x_pred - x_true)**2).sum(dim=-1) 
        
        #Orientation loss 
        n_pred_hat = F.normalize(n_pred, dim=-1, eps=1e-8)
        dot = (n_true * n_pred_hat).sum(dim =-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        n_loss = torch.arccos(dot)**2
        
        #Total Loss
        loss = (x_loss + self.w * n_loss).mean() 

        return loss

@torch.no_grad()
def pose_errors(pred : torch.Tensor,
                true : torch.Tensor
                ) -> tuple[torch.Tensor,torch.Tensor]:
    '''
    Per sample position and orientation error.

    Parameters
    ---------- 
    pred : torch.Tensor
        Predicted pose
    true : torch.Tensor 
        True Pose

    Returns 
    ------- 
    e_x, e_n : torch.Tensor,torch.Tensor
        Position and angle errors (mm, deg)
    '''

    if pred.shape[-1] == 5:
        pred = angles_to_normal_t(pred)
    if true.shape[-1] == 5:
        true = angles_to_normal_t(true)

    x_pred = pred[...,:3]
    n_pred = pred[...,3:]

    x_true = true[...,:3]
    n_true = true[...,3:]

    e_x = torch.linalg.norm(x_pred - x_true, dim = -1) * 1000.0
    
    #was originally using arccos but ran into issues where I had to clamp the dot which meant that the angular error had a floor of 0.0256 degrees
    n_pred_hat = F.normalize(n_pred, dim=-1, eps=1e-8)
    dot = (n_true * n_pred_hat).sum(dim =-1)
    cross = torch.linalg.cross(n_true,n_pred_hat,dim=-1).norm(dim=-1)
    e_n = 180/np.pi * torch.atan2(cross,dot)
    
    return e_x, e_n
        

def train(model : nn.Module,
          train_loader,
          test_loader,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.LRScheduler = None,
          epochs : int = 1000,
          loss_fn : nn.Module = None,
          device : torch.device = "cpu",
          print_losses : bool = True,
          log_path : str = None
          ) -> list[dict]: 
    '''
        General train loop for models. 
        
        Parameters
        ---------- 
        model : nn.Module 
            Model Class
        train_loader 
            Training Dataloder 
        test_loader 
            Test set dataloader
        optimizer : torch.optim.Optimizer
            Optimizer used for training
        scheduler : torch.optim.scheduler
            LR scheduler
        epochs : int 
            Number of epochs, default = 1000
        loss_fn : nn.Module
            Loss function, default = MSE 
        device : torch.device
            Device to run training on, default = "cpu"
        print_losses : bool 
            Print losses at each epoch, default = True
        log_path : str 
            If given, one record per epoch is appended to the file at specified path
        returns 
        ------- 
        history : list[dict]
            Summary statistics at each epoch
    '''

    if loss_fn is None:
        loss_fn = nn.MSELoss()

    log_file = open(log_path,"a") if log_path is not None else None
    history = []

    model = model.to(device)

    if print_losses:
        print("epoch \t LR \t Training Loss \t Test Loss \t e_p test \t e_n test")
        
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        for x, y in train_loader:

            x,y= x.to(device),y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred,y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
                


        model.eval()
        epoch_test_loss = 0.0
        epoch_e_p = 0.0
        epoch_e_n = 0.0

        with torch.no_grad():
            for x,y in test_loader:
                x,y = x.to(device), y.to(device)
                pred = model(x)
                epoch_test_loss += loss_fn(pred,y).item()
                e_p, e_n = pose_errors(pred,y)
                epoch_e_p += e_p.mean().item()
                epoch_e_n += e_n.mean().item()

        epoch_train_loss /= len(train_loader)
        epoch_test_loss /= len(test_loader)
        epoch_e_p /= len(test_loader)
        epoch_e_n /= len(test_loader)
        

        if print_losses:
            print(f"{epoch + 1} \t {optimizer.param_groups[0]['lr']:.2e} \t {epoch_train_loss:.4f} \t  {epoch_test_loss:.4f} \t {epoch_e_p:.4f} \t {epoch_e_n:.4f}")

        record = {
            "epoch": epoch + 1,
            "lr" : optimizer.param_groups[0]['lr'],
            "train_loss": epoch_train_loss,
            "test_loss": epoch_test_loss,
            "e_p_mean" : epoch_e_p,
            "e_n_mean" : epoch_e_n
        }
        history.append(record)
 
        if log_file is not None:
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()

        if scheduler is not None:
            scheduler.step()

    return history
