# utils/eval.py
import torch
import torch.nn.functional as F

def evaluate(model, dataloader, device):
    model.eval()
    test_loss = 0
    correct = 0
    total_samples = len(dataloader.dataset)

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Compute loss
            loss = F.cross_entropy(output, target, reduction='sum')
            test_loss += loss.item()
            
            # Get predictions
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= total_samples
    accuracy = 100. * correct / total_samples

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total_samples} ({accuracy:.2f}%)\n')
    
    return test_loss, accuracy
