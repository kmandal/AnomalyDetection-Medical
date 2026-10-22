import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from LinVAE_model import MedicalVAE
from ConvVAE_model import MedicalConvVAE

def train_and_evaluate(model_type, latent_dim, beta, dataloader, normal_test_tensor, anomaly_test_tensor, epochs=15):
    # 1. Initialize the correct model architecture
    if model_type == 'linear':
        model = MedicalVAE(input_dim=2352, latent_dim=latent_dim, beta=beta)
    else:
        model = MedicalConvVAE(latent_dim=latent_dim, beta=beta)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 2. Training Pass
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            x_recon, mu, logvar = model(x)
            loss = model.loss_function(x, x_recon, mu, logvar)
            loss.backward()
            optimizer.step()
            
    # 3. Evaluation Pass
    model.eval()
    with torch.no_grad():
        # Evaluate Normal Test
        x_recon_norm, _, _ = model(normal_test_tensor)
        norm_bce = F.binary_cross_entropy(x_recon_norm, normal_test_tensor, reduction='none').reshape(normal_test_tensor.size(0), -1).mean(dim=1)
        mean_normal = torch.mean(norm_bce).item()
        
        # Evaluate Anomaly Test
        x_recon_anom, _, _ = model(anomaly_test_tensor)
        anom_bce = F.binary_cross_entropy(x_recon_anom, anomaly_test_tensor, reduction='none').reshape(anomaly_test_tensor.size(0), -1).mean(dim=1)
        mean_anomaly = torch.mean(anom_bce).item()
        
    separation_indicator = mean_anomaly / mean_normal
    return mean_normal, mean_anomaly, separation_indicator


def main():
    parser = argparse.ArgumentParser(description="Tune Medical VAE Hyperparameters")
    parser.add_argument('--model', type=str, default='conv', choices=['linear', 'conv'], help="Model type to optimize")
    args = parser.parse_args()

    data_dir = "data"
    x_train = np.load(os.path.join(data_dir, "normal_train.npy"))
    normal_test = np.load(os.path.join(data_dir, "normal_test.npy"))
    anomaly_test = np.load(os.path.join(data_dir, "anomaly_test.npy"))
    
    # Apply structural layout transformations upfront
    if args.model == 'linear':
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32).view(x_train.shape[0], -1)
        normal_test_tensor = torch.tensor(normal_test, dtype=torch.float32).view(normal_test.shape[0], -1)
        anomaly_test_tensor = torch.tensor(anomaly_test, dtype=torch.float32).view(anomaly_test.shape[0], -1)
    else:
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2).contiguous()
        normal_test_tensor = torch.tensor(normal_test, dtype=torch.float32).permute(0, 3, 1, 2).contiguous()
        anomaly_test_tensor = torch.tensor(anomaly_test, dtype=torch.float32).permute(0, 3, 1, 2).contiguous()
        
    dataloader = DataLoader(TensorDataset(x_train_tensor), batch_size=128, shuffle=True)
    
    # 🎯 Tuning Grid: Tighten the bottleneck limits and expand structural penalties
    latent_dims = [2, 4, 8]
    betas = [1.0, 4.0, 10.0, 20.0]
    
    print(f"\n========================================================")
    print(f" Tuning Search Grid: {args.model.upper()} VAE")
    print(f"========================================================")
    print(f"{'Latent Dim':<12}{'Beta':<8}{'Norm Error':<12}{'Anom Error':<12}{'Separation':<12}")
    print("-" * 56)
    
    best_separation = 0.0
    best_params = {}
    
    for lat_dim in latent_dims:
        for b in betas:
            norm_err, anom_err, sep_ind = train_and_evaluate(
                model_type=args.model,
                latent_dim=lat_dim,
                beta=b,
                dataloader=dataloader,
                normal_test_tensor=normal_test_tensor,
                anomaly_test_tensor=anomaly_test_tensor,
                epochs=15
            )
            
            print(f"{lat_dim:<12}{b:<8.1f}{norm_err:<12.5f}{anom_err:<12.5f}{sep_ind:<12.2f}x")
            
            if sep_ind > best_separation:
                best_separation = sep_ind
                best_params = {'latent_dim': lat_dim, 'beta': b}
                
    print("-" * 56)
    print(f"🏆 Best {args.model.upper()} Config -> Latent Dim: {best_params['latent_dim']}, Beta: {best_params['beta']}")
    print(f"📈 Highest Achieved Separation: {best_separation:.2f}x")

if __name__ == "__main__":
    main()
