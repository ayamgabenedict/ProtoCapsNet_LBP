# test_pipeline.py
import torch
from models.proto_capsule_net import ProtoCapsuleNet
from utils.train import train
from utils.eval import evaluate
from torch.utils.data import DataLoader, TensorDataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create dummy dataset
num_samples = 20
X_dummy = torch.rand(num_samples, 1, 28, 28)  # Example grayscale images
y_dummy = torch.randint(0, 10, (num_samples,))  # 10-class labels

# DataLoader
dataset = TensorDataset(X_dummy, y_dummy)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize model
model = ProtoCapsuleNet(num_classes=10, num_prototypes=10).to(device)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Run a single training step to check for errors
print("Testing Training...")
train(model, dataloader, optimizer, epoch=1, device=device)

# Run evaluation step
print("Testing Evaluation...")
test_loss, accuracy = evaluate(model, dataloader, device)

print(f"Test completed. Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
