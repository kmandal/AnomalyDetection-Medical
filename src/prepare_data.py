import os
import numpy as np
import medmnist
from medmnist import INFO

def prepare_medmnist_data(dataset_name="pathmnist", data_dir="data"):
    print(f"Fetching MedMNIST dataset: {dataset_name}...")
        
    # Get dataset info
    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info['python_class'])
    
    # Download and load training and test splits
    train_dataset = DataClass(split='train', download=True, root=data_dir)
    test_dataset = DataClass(split='test', download=True, root=data_dir)
    
    # Extract images and labels
    x_train = train_dataset.imgs   # Shape: (N, 28, 28) for 28x28 datasets
    y_train = train_dataset.labels.squeeze()
    
    x_test = test_dataset.imgs
    y_test = test_dataset.labels.squeeze()
    
    print(f"Loaded {len(x_train)} training samples and {len(x_test)} test samples.")
    print(f"Dataset classes: {info['label']}" )
    
    # For Unsupervised Anomaly Detection:
    target_normal = 6 
    
    print(f"\nFiltering data: Treating class '{info['label'][str(target_normal)]}' as Normal background.")
    
    # Normal training data (Healthy baseline)
    normal_train_mask = (y_train == target_normal)
    x_train_normal = x_train[normal_train_mask]
    
    # Test data: Mix of normal and anomalies
    x_test_normal = x_test[y_test == target_normal]
    x_test_anomaly = x_test[(y_test == 7) | (y_test == 8)] # for specific targeted labels
    
    x_train_normal = x_train_normal.astype(np.float32) / 255.0
    x_test_normal = x_test_normal.astype(np.float32) / 255.0
    x_test_anomaly = x_test_anomaly.astype(np.float32) / 255.0

    # Save processed arrays
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, "normal_train.npy"), x_train_normal)
    np.save(os.path.join(data_dir, "normal_test.npy"), x_test_normal)
    np.save(os.path.join(data_dir, "anomaly_test.npy"), x_test_anomaly)
    
    print(f"Successfully saved processed arrays to {data_dir}/")
    print(f" - Normal Training samples: {len(x_train_normal)}")
    print(f" - Normal Test samples:     {len(x_test_normal)}")
    print(f" - Anomaly Test samples:    {len(x_test_anomaly)}")

if __name__ == "__main__":
    prepare_medmnist_data(dataset_name="pathmnist")
