import torch
import torch.nn as nn
import torch.nn.functional as F

class MedicalConvVAE(nn.Module):
    def __init__(self, latent_dim=4, beta=4.0):
        super(MedicalConvVAE, self).__init__()
        self.beta = beta
        
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=2),  # Out: (Batch, 8, 14, 14)
            nn.BatchNorm2d(8),
            nn.GELU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # Out: (Batch, 16, 7, 7)
            nn.BatchNorm2d(16),
            nn.GELU()
        )
        
        # Bottleneck Linear Layers (Flattened size is  16 * 7 * 7 = 784)
        self.fc_mu = nn.Linear(784, latent_dim)
        self.fc_logvar = nn.Linear(784, latent_dim)
        
        # 2. TUNED DECODER BLUEPRINT
        self.decoder_fc = nn.Linear(latent_dim, 784)
        
        # Transposed Convolutions expanding back up from (Batch, 16, 7, 7)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1), # Out: (Batch, 8, 14, 14)
            nn.BatchNorm2d(8),
            nn.GELU(),
            nn.ConvTranspose2d(8, 3, kernel_size=5, stride=2, padding=2, output_padding=1),  # Out: (Batch, 3, 28, 28)
            nn.Sigmoid() 
        )
        
    def encode(self, x):
        h = self.encoder_conv(x)
        h = h.reshape(h.size(0), -1) 
        return self.fc_mu(h), self.fc_logvar(h)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.reshape(h.size(0), 16, 7, 7) 
        return self.decoder_conv(h)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
        
    def loss_function(self, x, x_recon, mu, logvar):
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum') / x.size(0)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + self.beta * kld_loss
