import torch
import torch.nn.functional as F


def loss_criterion(inputs, targets, mu, log_var):
    # Reconstruction loss
    bce_loss = F.binary_cross_entropy(inputs, targets, reduction="sum")
    # Regularization term
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return bce_loss, kl_loss