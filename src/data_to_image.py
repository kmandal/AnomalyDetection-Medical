import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from medmnist import INFO

#these labels are for pathmnist dataset, for other types, check the info labels first
label_info = {
    '0': 'adipose', 
    '1': 'background', 
    '2': 'debris', 
    '3': 'lymphocytes', 
    '4': 'mucus', 
    '5': 'smooth muscle', 
    '6': 'normal colon mucosa', 
    '7': 'cancer-associated stroma', 
    '8': 'colorectal adenocarcinoma epithelium'
}


def plot_medmnist_data(dataset_name="pathmnist", data_dir="data", output_dir="output_files"):
    # Print the label info dynamically using info['label']
    info = INFO[dataset_name]
    label_dict = info['label']
    
    print("--- Dataset Label Info ---")
    for key, value in label_dict.items():
        print(f"Label {key}: {value}")
    print("--------------------------\n")


    npz_filename = f"{dataset_name}.npz"
    data_file = Path(data_dir) / npz_filename

    if not data_file.exists():
        print(f"⚠️ dataset file {npz_filename} not found at {data_dir}, download the dataset first.")
        return

    # Load data
    data = np.load(data_file)

    # Extract the training images and labels
    images = data['train_images']
    labels = data['train_labels'].flatten()

    # Find the first index belonging to Healthy Mucosa (6) and Cancer (8)
    healthy_idx = np.where(labels == 6)[0][0]
    cancer_idx = np.where(labels == 8)[0][0]

    # Plot them side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(images[healthy_idx])
    axes[0].set_title(f"Label 6: {label_dict['6']}")
    axes[0].axis('off')

    axes[1].imshow(images[cancer_idx])
    axes[1].set_title(f"Label 8:  {label_dict['8']}")
    axes[1].axis('off')
    
    plt.show()

    # Create output directories for saving files
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # extract and save multiple images (e.g., the first 3 of each)
    target_labels = [6, 8] 
    
    for lbl in target_labels:
        # Get an array of all indices that match this label
        all_matching_indices = np.where(labels == lbl)[0]
        
        # Take the first 3 matching indices
        sample_indices = all_matching_indices[:3] 
        
        for rank, idx in enumerate(sample_indices):
            raw_img = images[idx]
            label_name = label_dict[str(lbl)].replace(" ", "_")
            
            # Convert numpy array to a PIL Image object
            pil_img = Image.fromarray(raw_img)
            
            # Save as PNG or JPG
            file_name = f"label_{lbl}_{label_name}_sample_{rank+1}.png"
            pil_img.save(out_path / file_name)
            print(f"Saved: {out_path / file_name}")


if __name__ == "__main__":
    plot_medmnist_data(dataset_name="pathmnist", data_dir="data", output_dir="output_files")
