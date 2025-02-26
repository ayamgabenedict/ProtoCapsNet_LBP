import argparse
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from utils.data_preprocessing import prepare_dataloader
from models.proto_capsule_net import ProtoCapsuleNet

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataset paths
DATASET_PATHS = {
    "birds": "datasets/birds",
    "cars": "datasets/cars"
}

def visualize_predictions(dataset, num_images=8):
    """Visualizes model predictions on a batch of test images."""
    print(f"📸 Visualizing predictions for {dataset} dataset...")

    # Load data
    _, test_loader = prepare_dataloader(DATASET_PATHS[dataset], batch_size=num_images)

    # Load model
    model = ProtoCapsuleNet(num_classes=10, num_prototypes=10).to(device)
    model.load_state_dict(torch.load(f"proto_capsnet_{dataset}.pth"))
    model.eval()

    # Get a batch of test images
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    # Make predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    # Convert images to displayable format
    images = images.cpu().numpy().squeeze()  # Remove batch dimension

    # Plot results
    fig, axes = plt.subplots(2, num_images // 2, figsize=(12, 6))
    fig.suptitle(f"ProtoCapsNet Predictions on {dataset} Dataset", fontsize=14)

    for i, ax in enumerate(axes.flat):
        img = images[i]
        ax.imshow(img, cmap="gray")
        ax.axis("off")

        actual_label = labels[i].item()
        predicted_label = predicted[i].item()

        title_color = "green" if actual_label == predicted_label else "red"
        ax.set_title(f"True: {actual_label}\nPred: {predicted_label}", color=title_color)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize ProtoCapsNet Predictions")
    parser.add_argument("--dataset", type=str, choices=["birds", "cars"], required=True, help="Dataset to use")
    parser.add_argument("--num_images", type=int, default=8, help="Number of images to display")

    args = parser.parse_args()

    visualize_predictions(args.dataset, args.num_images)
