import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils.data_preprocessing import prepare_dataloader
from models.proto_capsule_net import ProtoCapsuleNet
import os

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataset paths
DATASET_PATHS = {
    "birds": "datasets/birds",
    "cars": "datasets/cars"
}

def train_model(dataset, epochs, batch_size, learning_rate):
    """Train ProtoCapsNet on the specified dataset."""
    print(f"🚀 Training ProtoCapsNet on {dataset} dataset...")

    # Load data
    train_loader, test_loader = prepare_dataloader(DATASET_PATHS[dataset], batch_size=batch_size)

    # Initialize model
    model = ProtoCapsuleNet(num_classes=10, num_prototypes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    
    # Ensure saved_models directory exists
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")
    
    # Save model
    torch.save(model.state_dict(), f"saved_models/proto_capsnet_{dataset}.pth")
    print(f"✅ Model saved as saved_models/proto_capsnet_{dataset}.pth")

    
    

def evaluate_model(dataset):
    """Evaluate the trained ProtoCapsNet model."""
    print(f"🔍 Evaluating ProtoCapsNet on {dataset} dataset...")

    # Load data
    _, test_loader = prepare_dataloader(DATASET_PATHS[dataset])

    # Load model
    model = ProtoCapsuleNet(num_classes=10, num_prototypes=10).to(device)
    model.load_state_dict(torch.load(f"proto_capsnet_{dataset}.pth"))
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"✅ Accuracy on {dataset} dataset: {accuracy:.2f}%")

def predict_image(image_path, dataset):
    """Make a prediction on a single image."""
    from PIL import Image
    from torchvision import transforms

    print(f"🖼️ Predicting image: {image_path} using ProtoCapsNet trained on {dataset} dataset...")

    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Load model
    model = ProtoCapsuleNet(num_classes=10, num_prototypes=10).to(device)
    model.load_state_dict(torch.load(f"proto_capsnet_{dataset}.pth"))
    model.eval()

    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)

    print(f"✅ Predicted class: {predicted_class.item()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, Evaluate, or Predict using ProtoCapsNet")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--predict", action="store_true", help="Predict a single image")
    parser.add_argument("--dataset", type=str, choices=["birds", "cars"], help="Dataset to use")
    parser.add_argument("--image_path", type=str, help="Path to the image for prediction")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training")

    args = parser.parse_args()

    if args.train and args.dataset:
        train_model(args.dataset, args.epochs, args.batch_size, args.lr)
    elif args.evaluate and args.dataset:
        evaluate_model(args.dataset)
    elif args.predict and args.image_path and args.dataset:
        predict_image(args.image_path, args.dataset)
    else:
        print("❌ Invalid command. Use --help for usage details.")
