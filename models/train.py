import torch
import torch.nn as nn
import numpy as np

def train(model : nn.Module,
          train_loader,
          test_loader,
          optimizer: torch.optim.Optimizer,
          epochs : int = 1000,
          loss_fn : nn.Module = None,
          device : torch.device = "cpu",
          print_losses : bool = True
          ) -> tuple[list[float],list[float]]: 
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
        epochs : int 
            Number of epochs, default = 1000
        loss_fn : nn.Module
            Loss function, default = MSE 
        device : torch.device
            Device to run training on, default = "cpu"
        print_losses : bool 
            Print losses at each epoch, default = True

        returns 
        ------- 
        train_losses, test_losses : list[float], list[float]
            Loss at each epoch
    '''

    if loss_fn is None:
        loss_fn = nn.MSELoss()

    train_losses = []
    test_losses = []

    model = model.to(device)
    
    if print_losses:
        print(f"epoch \t Training Loss \t Test Loss \t Mean pos error train \t Mean Pos Error Test")
        
    for epoch in range(epochs):
        epoch_train_loss = 0
        epoch_train_pos_loss = 0
        model.train()
        for x, y in train_loader:

            x,y= x.to(device),y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred,y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            epoch_train_pos_loss += (y[:, :3] - pred[:, :3]).norm(dim=1).mean().item() 
                
        with torch.no_grad():
            model.eval()
            epoch_test_loss = 0 
            epoch_test_pos_loss = 0
            for x,y in test_loader:

                x,y = x.to(device), y.to(device)
                pred = model(x)
                epoch_test_loss += loss_fn(pred,y).item()
                epoch_test_pos_loss += (y[:, :3] - pred[:, :3]).norm(dim=1).mean().item()

        epoch_train_loss /= len(train_loader)
        epoch_train_pos_loss /= len(train_loader)
        epoch_test_loss /= len(test_loader)
        epoch_test_pos_loss /= len(test_loader)
        
        if print_losses:
            print(f"{epoch + 1} \t {epoch_train_loss:.4f} \t  {epoch_test_loss:.4f} \t {epoch_train_pos_loss:.4f} \t {epoch_test_pos_loss:.4f}")

        train_losses.append(epoch_train_loss)
        test_losses.append(epoch_test_loss)


    return train_losses, test_losses
