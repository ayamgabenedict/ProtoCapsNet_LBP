import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split

# Define dataset paths
BIRDS_DATASET_PATH = "datasets/birds"
CARS_DATASET_PATH = "datasets/cars"

# Ensure dataset directories exist
if not os.path.exists(BIRDS_DATASET_PATH) or not os.path.exists(CARS_DATASET_PATH):
    raise FileNotFoundError("Dataset directories not found! Please download and extract the datasets.")

# Define LBP parameters
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, lbp_enabled=True):
        self.root_dir = root_dir
        self.transform = transform
        self.lbp_enabled = lbp_enabled
        self.image_paths = []
        self.labels = []

        self._load_images()

    def _load_images(self):
        """Loads image paths and corresponding labels from dataset directory"""
        class_folders = sorted(os.listdir(self.root_dir))
        for class_idx, class_name in enumerate(class_folders):
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_path):
                for image_name in os.listdir(class_path):
                    if image_name.endswith(('.jpg', '.jpeg', '.png')):  
                        self.image_paths.append(os.path.join(class_path, image_name))
                        self.labels.append(class_idx)

    def _apply_lbp(self, img):
        """Apply Local Binary Pattern (LBP) transformation."""
        lbp = cv2.LBP(img, LBP_POINTS, LBP_RADIUS, method="uniform")
        return lbp.astype(np.float32) / 255.0  # Normalize

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Error loading image: {img_path}")

        # Apply LBP if enabled
        if self.lbp_enabled:
            img = self._apply_lbp(img)

        # Convert to tensor and apply transformations
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        if self.transform:
            img = self.transform(img)

        return img, label

def prepare_dataloader(dataset_path, batch_size=32, train_split=0.8):
    """Prepares DataLoader with train/test split."""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize for uniformity
        transforms.ToTensor(),
    ])
    
    dataset = ImageDataset(dataset_path, transform=transform)

    # Stratified train/test split
    train_indices, test_indices = train_test_split(
        np.arange(len(dataset)), test_size=1-train_split, stratify=dataset.labels, random_state=42
    )

    train_set = torch.utils.data.Subset(dataset, train_indices)
    test_set = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    print("Preparing data loaders for Birds and Cars datasets...")

    birds_train_loader, birds_test_loader = prepare_dataloader(BIRDS_DATASET_PATH)
    cars_train_loader, cars_test_loader = prepare_dataloader(CARS_DATASET_PATH)

    print(f"✅ Birds Dataset - Train: {len(birds_train_loader.dataset)}, Test: {len(birds_test_loader.dataset)}")
    print(f"✅ Cars Dataset - Train: {len(cars_train_loader.dataset)}, Test: {len(cars_test_loader.dataset)}")
    print("Data preprocessing complete! 🚀")
