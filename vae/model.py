import torch
import torch.nn as nn

# Flow: Encoder -> LatentZ -> Decoder

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mu = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        return self.fc(x)

class LatentZ(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(input_dim, hidden_dim)
        self.z = LatentZ(hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, log_var):
        std = (0.5 * log_var).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, train_sample: bool = True):
        h = self.encoder(x)
        mu, log_var = self.z(h)

        if train_sample:
            z = self.reparameterize(mu, log_var) # (training)
        else:
            z = mu                          # deterministic (evaluation)

        recon = self.decoder(z)

        return recon, mu, log_var, z