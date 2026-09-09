import numpy as np
import torch
import torch.nn as nn
from tqdm.notebook import tqdm

class Trainer(nn.Module):
    def __init__(self, model, optimizer, loss_fn, device):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def run_training(self, train_loader, val_loader, epochs):
        self.model.to(self.device)

        train_losses = []
        val_losses = []
        train_recon_losses = []
        train_kl_losses = []

        epoch_step = 0
        global_step = 0

        for epoch in tqdm(range(epochs)):
            # TRAINING
            self.model.train()
            running_loss = 0
            running_recon_loss = 0
            running_kl_loss = 0
            n_batches = len(train_loader)

            for x, _ in train_loader:
                x = x.view(x.shape[0], -1).float()
                x = (x > 0.5).float().to(self.device)

                recon, mu, log_var, _ = self.model(x)
                bce_loss, kl_loss = self.loss_fn(recon, x, mu, log_var)

                loss = bce_loss + kl_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss
                running_recon_loss += bce_loss
                running_kl_loss += kl_loss
                global_step += global_step

            epoch_step = epoch + 1
            epoch_train_loss = (running_loss / n_batches).item()
            epoch_train_recon_loss = (running_recon_loss / n_batches).item()
            epoch_train_kl_loss = (running_kl_loss / n_batches).item()

            train_losses.append(epoch_train_loss)
            train_recon_losses.append(epoch_train_recon_loss)
            train_kl_losses.append(epoch_train_kl_loss)

            # VALIDATION
            val_loss = self.run_validation(val_loader)
            val_losses.append(val_loss)

        return np.array(train_losses), np.array(val_losses), np.array(train_recon_losses), np.array(train_kl_losses), epoch_step, global_step

    def run_validation(self, dataloader):
        self.model.eval()
        running_loss = 0
        n_batches = len(dataloader)

        with torch.no_grad():
            for x, _ in dataloader:
                x = x.view(x.shape[0], -1).float()
                x = (x > 0.5).float().to(self.device)

                recon, mu, log_var, _ = self.model(x)
                bce_loss, kl_loss = self.loss_fn(recon, x, mu, log_var)
                loss = bce_loss + kl_loss
                running_loss += loss

        return (running_loss / n_batches).item()