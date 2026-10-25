"""
Stage 1: Train a small, fully-inspectable CNN on FashionMNIST.

Design choices made on purpose for the later fault-injection work:
- Small architecture: every weight tensor is easy to enumerate and flip bits in.
- No batchnorm: keeps weight->output relationship simpler to reason about
  when you start injecting faults (batchnorm running stats add a second,
  harder-to-interpret failure mode).
- Saves both a full checkpoint (for resuming/inspecting) and a clean
  "golden" copy of state_dict you'll diff fault-injected models against.
"""

# PyTorch's main package. It provides tensors (arrays used by the model),
# model saving/loading, and GPU support.
import torch

# Neural-network layers such as Conv2d, Linear, and MaxPool2d.  We use the
# short name "nn" because it is the conventional PyTorch style.
import torch.nn as nn

# Neural-network operations that are functions rather than stored layers.
# This script uses F.relu(), which changes negative values to zero.
import torch.nn.functional as F

# DataLoader supplies the dataset to the model in small groups (batches)
# instead of loading all images into memory at once.
from torch.utils.data import DataLoader

# torchvision provides ready-to-use image datasets and image preprocessing
# steps. We use FashionMNIST from datasets and tensor/normalization steps from
# transforms.
from torchvision import datasets, transforms

# Use an NVIDIA GPU through CUDA when PyTorch can access one; otherwise use
# the CPU. Later, both the model and image tensors are moved to this device.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of images processed together before the model updates its weights.
# A batch of 128 is a reasonable size for this small dataset.
BATCH_SIZE = 128

# One epoch means that the model has seen every training image once.
EPOCHS = 8

# The learning rate controls how large each weight update is during training.
# 1e-3 is scientific notation for 0.001.
LR = 1e-3

# File where the trained model weights and final test accuracy are saved.
CKPT_PATH = "fashionmnist_cnn.pt"


class SimpleCNN(nn.Module):
    """Two conv blocks + two FC layers. Small enough to enumerate every
    weight tensor by name for bit-flip experiments later."""

    def __init__(self, num_classes: int = 10):
        # nn.Module initializes the PyTorch model machinery. Every custom
        # neural-network model that inherits from nn.Module needs this call.
        super().__init__()

        # A FashionMNIST image is grayscale, so it has 1 input channel.
        # conv1 learns 16 small 3x3 filters (for simple patterns such as edges).
        # padding=1 keeps the image width and height at 28x28.
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # 1x28x28 -> 16x28x28

        # The second convolution combines the first layer's patterns into
        # 32 more detailed feature maps.
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 16x14x14 -> 32x14x14

        # Max pooling keeps the largest value in each 2x2 region. This halves
        # the width and height while retaining strong detected features.
        self.pool = nn.MaxPool2d(2, 2)

        # After two pooling steps: 32 feature maps, each of size 7x7.
        # fc1 combines those 1,568 values into 128 learned features.
        self.fc1 = nn.Linear(32 * 7 * 7, 128)

        # One output score (a logit) for each clothing class: 10 in total.
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Each image flows through the layers in this order. ReLU adds
        # non-linearity: it replaces negative values with zero.
        x = self.pool(F.relu(self.conv1(x)))   # 1x28x28 -> 16x14x14
        x = self.pool(F.relu(self.conv2(x)))   # 16x14x14 -> 32x7x7

        # Flatten changes each image's 32x7x7 features into one vector of
        # 1,568 numbers, keeping the batch dimension unchanged.
        x = torch.flatten(x, 1)

        # The first fully connected layer learns from all extracted features.
        x = F.relu(self.fc1(x))

        # The final layer produces 10 logits. Softmax is intentionally not
        # applied here because CrossEntropyLoss applies the needed operation
        # internally in a numerically stable way.
        x = self.fc2(x)
        return x 


def get_dataloaders():
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),  # FashionMNIST mean/std
    ])
    train_set = datasets.FashionMNIST(root="./data", train=True, download=True, transform=tfm)
    test_set = datasets.FashionMNIST(root="./data", train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)
    return train_loader, test_loader
    model.eval()  
    


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def train():
    train_loader, test_loader = get_dataloaders()
    model = SimpleCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        test_acc = evaluate(model, test_loader)
        print(f"epoch {epoch:2d}/{EPOCHS}  train_loss={train_loss:.4f}  test_acc={test_acc:.4f}")

    # Save everything you'll need later: architecture args, weights, and
    # a clean baseline accuracy to compare fault-injected accuracy against.
    torch.save({
        "model_state_dict": model.state_dict(),
        "test_acc": evaluate(model, test_loader),
    }, CKPT_PATH)
    print(f"saved checkpoint to {CKPT_PATH}")


if __name__ == "__main__":
    train()
