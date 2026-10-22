import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np

from LinVAE_model import MedicalVAE
from ConvVAE_model import MedicalConvVAE

def main():
    parser = argparse.ArgumentParser(description="Evaluate Medical VAE Anomaly Scores")
    parser.add_argument('--model', type=str, default='conv', choices=['linear', 'conv'])
    args = parser.parse_args()

    data_dir = "data"
    normal_test = np.load(os.path.join(data_dir, "normal_test.npy"))
    anomaly_test = np.load(os.path.join(data_dir, "anomaly_test.npy"))
    
    # 1. Transform inputs identically to how training was structured
    if args.model == 'linear':
        norm_tensor = torch.tensor(normal_test, dtype=torch.float32).view(normal_test.shape[0], -1)
        anom_tensor = torch.tensor(anomaly_test, dtype=torch.float32).view(anomaly_test.shape[0], -1)
        model = MedicalVAE(input_dim=2352, latent_dim=16, beta=1.0)
    else:
        norm_tensor = torch.tensor(normal_test, dtype=torch.float32).permute(0, 3, 1, 2).contiguous()
        anom_tensor = torch.tensor(anomaly_test, dtype=torch.float32).permute(0, 3, 1, 2).contiguous()
        model = MedicalConvVAE(latent_dim=16, beta=1.0)

    # Load respective weight checkpoint
    model.load_state_dict(torch.load(os.path.join(data_dir, f"medical_{args.model}_vae.pth"), weights_only=True))
    model.eval()
    
    with torch.no_grad():
        # Evaluate Normal Controls
        x_recon_norm, _, _ = model(norm_tensor)
        # Using .view(..., -1) inside evaluation ensures error is averaged across features cleanly for both structures
        norm_bce = F.binary_cross_entropy(x_recon_norm, norm_tensor, reduction='none').view(norm_tensor.size(0), -1).mean(dim=1)
        mean_normal = torch.mean(norm_bce).item()
        
        # Evaluate Anomalies
        x_recon_anom, _, _ = model(anom_tensor)
        anom_bce = F.binary_cross_entropy(x_recon_anom, anom_tensor, reduction='none').view(anom_tensor.size(0), -1).mean(dim=1)
        mean_anomaly = torch.mean(anom_bce).item()
        
    print(f"\n[{args.model.upper()} Evaluation Results]")
    print(f"Mean Normal Test Reconstruction Error:  {mean_normal:.6f}")
    print(f"Mean Cancer Tissue Reconstruction Error: {mean_anomaly:.6f}")
    print(f"-> Separation indicator: Cancer score is {mean_anomaly / mean_normal:.2f}x higher than normal.")

if __name__ == "__main__":
    main()
