import os
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.optim as optim

from LinVAE_model import MedicalVAE
from ConvVAE_model import MedicalConvVAE

def main():
    parser = argparse.ArgumentParser(description="Train Medical VAEs")
    parser.add_argument('--model', type=str, default='conv', choices=['linear', 'conv'], help="Choose model type: 'linear' or 'conv'")
    args = parser.parse_args()

    data_dir = "data"
    x_train = np.load(os.path.join(data_dir, "normal_train.npy")) # Shape: (N, 28, 28, 3)
    
    # 1. Dynamic Layout Transformation based on the model choice
    if args.model == 'linear':
        # Flatten image data for LinVAE -> (N, 2352)
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32).view(x_train.shape[0], -1)
        model = MedicalVAE(input_dim=2352, latent_dim=16, beta=1.0)
    else:
        # Permute to Channel-First layout for ConvVAE -> (N, 3, 28, 28)
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2)
        model = MedicalConvVAE(latent_dim=16, beta=1.0)

    dataloader = DataLoader(TensorDataset(x_train_tensor), batch_size=128, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"Training {args.model.upper()} VAE on healthy colon mucosa...")
    model.train()
    
    for epoch in range(15):
        total_loss = 0.0
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            x_recon, mu, logvar = model(x)
            loss = model.loss_function(x, x_recon, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/15 - Loss: {total_loss / len(dataloader):.4f}")
        
    # Save with unique file names based on selection
    torch.save(model.state_dict(), os.path.join(data_dir, f"medical_{args.model}_vae.pth"))
    print(f"Model saved as medical_{args.model}_vae.pth")

if __name__ == "__main__":
    main()
