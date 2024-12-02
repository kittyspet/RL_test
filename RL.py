import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchmetrics import Accuracy


# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(2)(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x


def get_cifar10_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False)
    return trainloader, testloader


def calculate_mnlp(predictions):
    log_probs = np.log(predictions + 1e-10)  # Avoid log(0)
    summed_log_probs = np.sum(log_probs, axis=1)
    normalized_log_probs = summed_log_probs / predictions.shape[1]
    return np.max(normalized_log_probs)


# Main routine
def main():
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.001

    trainloader, testloader = get_cifar10_dataloaders(batch_size)

    model = SimpleCNN().cuda()  # Move model to GPU
    criterion = nn.CrossEntropyLoss()

    # Initialize accuracy metric for multiclass classification
    accuracy_metric = Accuracy(task="multiclass", num_classes=10).cuda()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        for images, labels in trainloader:
            images, labels = images.cuda(), labels.cuda()  # Move data to GPU

            optimizer.zero_grad()  # Zero gradients
            outputs = model(images)  # Forward pass

            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item()

            # Calculate accuracy for this batch
            acc = accuracy_metric(outputs.softmax(dim=-1), labels)  # Use softmax on outputs

        # Average loss for the epoch
        avg_loss = running_loss / len(trainloader)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}')

    # Evaluate on test set and calculate MNLP
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for images, labels in testloader:
            images = images.cuda()
            outputs = model(images)
            probabilities = nn.Softmax(dim=1)(outputs).cpu().numpy()
            all_predictions.extend(probabilities)

    all_predictions = np.array(all_predictions)

    # Calculate MNLP score
    mnlp_score = calculate_mnlp(all_predictions)
    print(f'Maximum Normalized Log-Probability (MNLP): {mnlp_score}')


if __name__ == '__main__':
    main()