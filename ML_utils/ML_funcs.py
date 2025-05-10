import numpy as np


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from IPython.display import clear_output   # Jupyter live‑update helper



# ====================================================================================
# TOPOLOGIES
# ====================================================================================

class MLP_v1(nn.Module):
    """
    Fully‑connected net : input 1000 S21_dB → output 10 LC values.

    Parameters
    ----------
    in_dim      : int            input size  (default 1000)
    hidden_dims : tuple[int,...] hidden sizes
    out_dim     : int            output size (default 10)
    act         : nn.Module      activation   (default ReLU)
    drop_prob   : float          dropout p in hidden layers (0.0 = off)
    """

    def __init__(self,
                 in_dim: int = 1000,
                 hidden_dims: tuple[int, ...] = (512, 256, 128),
                 out_dim: int = 10,
                 act: nn.Module = nn.ReLU(),
                 drop_prob: float = 0.0):
        super().__init__()

        dims = (in_dim, ) + hidden_dims + (out_dim, )
        layers = []

        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act)
            if drop_prob > 0.0:
                layers.append(nn.Dropout(drop_prob))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)






# ====================================================================================
# Custom Loss Function 
# ====================================================================================


class MSPELoss(nn.Module):
    """
    Mean Squared Percentage Error
      L = mean( ((y_pred - y_true) / (y_true + eps)) ** 2 )
    eps avoids divide‑by‑zero.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        perc = (pred - target) / (target + self.eps)
        return torch.mean(perc ** 2)






# ====================================================================================
# Data Loader
# ====================================================================================


def create_dataloader(s21_data: np.ndarray, filter_param: np.ndarray, batch_size: int=128, shuffle: bool=True) -> DataLoader:

    # Convert numpy to tensor
    s21_data_tensor = torch.from_numpy( s21_data )
    filter_param_tensor = torch.from_numpy( filter_param )

    dataset = TensorDataset( s21_data_tensor, filter_param_tensor )  # inputs, outputs
    
    return DataLoader( dataset=dataset, batch_size=batch_size, shuffle=shuffle )





# ====================================================================================
# Training Functions
# ====================================================================================




def train_epoch(model, dataloader, criterion, optimizer, device):

    model.train()   # puts model in training mode

    total_loss = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)   # load batch into GPU
        optimizer.zero_grad()                       # clear gradients from last batch
        pred = model(x)                             # runs inference on full batch
        loss = criterion(pred, y)                   # computes loss of full batch
        loss.backward()                             # back prop
        optimizer.step()                            # adjust weights
        total_loss += loss.item() * x.size(0)       # adds to total_loss

    return total_loss / len(dataloader.dataset)     # retunrs avg loss





@torch.no_grad  # speeds up comp
def eval_epoch(model, dataloader, criterion, device):

    model.eval()            # eval mode
    total_loss = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)   # load batch into GPU
        pred = model(x)                             # runs inference on full batch
        loss = criterion(pred, y)                   # computes loss of full batch
        total_loss += loss.item() * x.size(0)       # adds to total_loss

    return total_loss / len(dataloader.dataset)     # retunrs avg loss





def train_model(train_dl, val_dl,
                criterion = nn.MSELoss(),
                batch_size: int = 128,
                lr: float = 1e-3,
                weight_decay: float = 1e-4,
                num_epochs: int = 50,
                drop_prob: float = 0.05,
                device: str = "cpu",
                plotting: bool = False):

    plot_every = 50


    model = MLP_v1(drop_prob=drop_prob).to(device)
    # train_dl.to(device)
    # val_dl.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


    plot_every   = 10          # redraw every N epochs
    hist_train   = []          # keep a running list of train losses
    hist_val     = []          # optional val losses
    fig, ax = plt.subplots(figsize=(6,4))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Training curve")
    line_tr, = ax.plot([], [], label="train")
    line_val, = ax.plot([], [], label="val")
    ax.legend()
    plt.show(block=False)
    plt.pause(0.1)
    ax.grid()



    for ep in range(1, num_epochs + 1):
        tr = train_epoch(model=model, dataloader=train_dl, criterion=criterion, optimizer=optimizer, device=device)
        
        hist_train.append(tr)
        
        if val_dl is not None:
            val = eval_epoch(model=model, dataloader=val_dl, criterion=criterion, device=device)
            hist_val.append(val)
            print(f"Epoch {ep:3d} | train {tr:.6f} | val {val:.6f}")
        else:
            print(f"Epoch {ep:3d} | train {tr:.6f}")


        if plotting and (ep % plot_every == 0 or ep == num_epochs):
            clear_output(wait=True)          # clears previous figure in notebook
            display(fig)
            line_tr.set_data(range(1, ep+1), hist_train)
            ax.relim(); ax.autoscale_view()
            ax.set_ylim(0, 5*val)
            if val_dl is not None:
                line_val.set_data(range(1, ep+1), hist_val)
            fig.canvas.draw()
            # plt.pause(0.1)


    return model













# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import lightning as pl

# class MLP_v1(pl.LightningModule):
    
#     # DEFINES TOPOLOGY
#     def __init__(self, input_dim=100, hidden_dim=1200, dropout_rate=0.2):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(hidden_dim, 1)  # Output: single value
#         )
#         self.train_loss = []
#         self.eval_loss = []
#         self.test_loss = []

#     # RUNS MODEL INFERENCE (only)
#     def forward(self, x):
#         return self.model(x)

#     # RUNS MODEL INFERENCE, COMPUTES LOSS, SAVES LOSS (training)
#     def training_step(self, batch, batch_idx):
#         x, y = batch                            # sets training inputs (x), and outputs (y)
#         y_hat = self(x)                         # runs inference and saves results
#         loss = F.mse_loss(y_hat.squeeze(), y)   # computes loss
#         self.train_loss.append( loss.item() )   # save training loss
#         # print(" -- train step")
#         return loss

#     # RUNS MODEL INFERENCE, COMPUTES LOSS, SAVES LOSS (validation)
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         val_loss = F.mse_loss(y_hat.squeeze(), y)
#         self.eval_loss.append( val_loss.item() )
#         # print(" --------------- eval step")
#         return val_loss

#     # RUNS MODEL INFERENCE, COMPUTES LOSS, SAVES LOSS (test)
#     def test_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         test_loss = F.mse_loss(y_hat.squeeze(), y)
#         self.test_loss.append( test_loss.item() )
#         return test_loss


#     # TELLS HOW TO UPDATE WEIGHTS
#     def configure_optimizers(self):
#         # return torch.optim.Adam(self.parameters(), lr=1e-3)
#         return torch.optim.Adam(self.parameters(), lr=0.5e-3)
#         # return torch.optim.SGD(self.parameters(), lr=0.05, momentum=0.8)