"""
Federated version: the SAME model and data, but split across several virtual
clients that train locally and are aggregated with FedAvg by a central server.
Uses Flower's simulation engine, so everything runs in one process on your machine.

This is the core artifact to show Prof. Wong: it demonstrates you actually ran
FedAvg and understand the client/server split.
"""
import torch
from collections import OrderedDict
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import flwr as fl
from flwr.simulation import start_simulation
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig

from centralized import Net, train, test  # reuse the same model + helpers

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLIENTS = 5
ROUNDS = 3


def load_datasets():
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST("./data", train=True, download=True, transform=tf)
    test_set = datasets.MNIST("./data", train=False, download=True, transform=tf)

    # Split the training set evenly across clients (IID split).
    n = len(train_set) // NUM_CLIENTS
    client_loaders = []
    for i in range(NUM_CLIENTS):
        idx = list(range(i * n, (i + 1) * n))
        client_loaders.append(DataLoader(Subset(train_set, idx), batch_size=64, shuffle=True))
    test_loader = DataLoader(test_set, batch_size=256)
    return client_loaders, test_loader


client_loaders, test_loader = load_datasets()


def get_params(model):
    return [v.cpu().numpy() for v in model.state_dict().values()]


def set_params(model, params):
    state = OrderedDict(
        {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), params)}
    )
    model.load_state_dict(state, strict=True)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, loader):
        self.model = Net().to(DEVICE)
        self.loader = loader

    def get_parameters(self, config):
        return get_params(self.model)

    def fit(self, parameters, config):
        set_params(self.model, parameters)
        train(self.model, self.loader, epochs=1)  # local training
        return get_params(self.model), len(self.loader.dataset), {}

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
        acc = test(self.model, test_loader)
        return 0.0, len(test_loader.dataset), {"accuracy": acc}


def client_fn(cid: str):
    return FlowerClient(client_loaders[int(cid)]).to_client()


def weighted_avg(metrics):
    accs = [num * m["accuracy"] for num, m in metrics]
    total = sum(num for num, _ in metrics)
    return {"accuracy": sum(accs) / total}


if __name__ == "__main__":
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_avg,
    )
    start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
    )
    print("Done. Check the 'accuracy' printed each round above.")
