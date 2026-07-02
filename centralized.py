"""
Baseline: train a small CNN on MNIST the normal (centralized) way.
Run this FIRST to get a reference accuracy, then compare with federated.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """A small CNN. Small on purpose so it trains fast on a laptop/Colab."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def load_data():
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST("./data", train=True, download=True, transform=tf)
    test = datasets.MNIST("./data", train=False, download=True, transform=tf)
    return DataLoader(train, batch_size=64, shuffle=True), DataLoader(test, batch_size=256)


def train(model, loader, epochs=1):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()


def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


if __name__ == "__main__":
    train_loader, test_loader = load_data()
    model = Net().to(DEVICE)
    for epoch in range(3):
        train(model, train_loader, epochs=1)
        acc = test(model, test_loader)
        print(f"[centralized] epoch {epoch+1}  accuracy = {acc:.4f}")
