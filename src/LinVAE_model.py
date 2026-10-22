import torch
import torch.nn as nn
import torch.nn.functional as F

class MedicalVAE(nn.Module):
    def __init__(self, input_dim=2352, latent_dim=16, beta=1.0): #28x28 image with 3 color channels, so 28 × 28 × 3 = 2352
        super(MedicalVAE, self).__init__()
        self.beta = beta
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, input_dim)
        
    def encode(self, x):
        h = F.gelu(self.fc1(x))
        h = F.gelu(self.fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        h = F.gelu(self.fc3(z))
        h = F.gelu(self.fc4(h))
        # Sigmoid bounds output pixels between 0 and 1 to match normalization
        return torch.sigmoid(self.fc5(h))
        
    def forward(self, x):
        x = x.view(x.size(0), -1) #Flatten x from (batch_size, height, width, channels) to (batch_size, input_dim)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
        
    def loss_function(self, x, x_recon, mu, logvar):
        # Reconstruction loss (using BCE for [0,1] pixel values)
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum') / x.size(0)
        # KL Divergence regularization term
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + self.beta * kld_loss
