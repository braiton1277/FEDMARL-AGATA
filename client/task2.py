
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

DEVICE = "cpu"

class Net(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        return self.fc2(x)

def make_server_val_balanced(
    targets,
    per_class: int = 200,
    n_classes: int = 10,
    seed: int = 0,
) -> List[int]:
    """
    Reserva uma quantidade balanceada de imagens de cada classe
    para a futura validação do servidor.
    """
    rng = np.random.RandomState(seed)

    label_to_idxs: Dict[int, List[int]] = {
        label: [] for label in range(n_classes)
    }

    for idx, target in enumerate(targets):
        label_to_idxs[int(target)].append(idx)

    validation_indices: List[int] = []

    for label in range(n_classes):
        class_indices = label_to_idxs[label]
        rng.shuffle(class_indices)
        validation_indices.extend(class_indices[:per_class])

    rng.shuffle(validation_indices)

    return validation_indices


def make_clients_dirichlet_indices(
    targets,
    train_pool_indices: List[int],
    n_clients: int = 50,
    alpha: float = 0.3,
    seed: int = 123,
    n_classes: int = 10,
) -> List[List[int]]:
    """
    Divide os índices do train_pool entre os clientes usando
    uma distribuição Dirichlet sobre as classes.
    """
    if n_clients <= 0:
        raise ValueError("n_clients deve ser maior que zero")

    if alpha <= 0:
        raise ValueError("alpha deve ser maior que zero")

    rng = np.random.RandomState(seed)

    label_to_idxs: Dict[int, List[int]] = {
        label: [] for label in range(n_classes)
    }

    for idx in train_pool_indices:
        label = int(targets[idx])
        label_to_idxs[label].append(int(idx))

    for label in range(n_classes):
        rng.shuffle(label_to_idxs[label])

    clients: List[List[int]] = [
        [] for _ in range(n_clients)
    ]

    for label in range(n_classes):
        class_indices = label_to_idxs[label]

        proportions = rng.dirichlet(
            alpha * np.ones(n_clients)
        )

        counts = (
            proportions * len(class_indices)
        ).astype(int)

        # Corrige a diferença causada pelo arredondamento.
        difference = len(class_indices) - counts.sum()

        if difference > 0:
            selected_clients = rng.choice(
                n_clients,
                size=difference,
                replace=True,
            )

            for client_id in selected_clients:
                counts[client_id] += 1

        elif difference < 0:
            available_clients = np.where(counts > 0)[0]

            selected_clients = rng.choice(
                available_clients,
                size=-difference,
                replace=True,
            )

            for client_id in selected_clients:
                counts[client_id] -= 1

        start = 0

        for client_id in range(n_clients):
            count = int(counts[client_id])

            if count > 0:
                clients[client_id].extend(
                    class_indices[start:start + count]
                )
                start += count

    for client_id in range(n_clients):
        rng.shuffle(clients[client_id])

    return clients


def load_data(
    data_path,
    client_id: int,
    n_clients: int = 50,
    alpha: float = 0.3,
    seed: int = 2048,
    val_per_class: int = 200,
    batch_size: int = 64,
    num_workers: int = 0,
):
    if not 0 <= client_id < n_clients:
        raise ValueError(
            f"CLIENT_ID inválido: {client_id}. "
            f"O valor deve estar entre 0 e {n_clients - 1}."
        )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        ),
    ])

    full_train = torchvision.datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=transform,
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True,
        transform=transform,
    )

    targets = np.asarray(
        full_train.targets,
        dtype=np.int64,
    )

    # No FEDMARL original:
    # seed da validação = SEED + 4242.
    server_val_indices = make_server_val_balanced(
        targets=targets,
        per_class=val_per_class,
        n_classes=10,
        seed=seed + 4242,
    )

    server_val_set = set(server_val_indices)

    train_pool_indices = [
        idx
        for idx in range(len(full_train))
        if idx not in server_val_set
    ]

    # No FEDMARL original:
    # seed da partição = SEED + 777.
    all_client_indices = make_clients_dirichlet_indices(
        targets=targets,
        train_pool_indices=train_pool_indices,
        n_clients=n_clients,
        alpha=alpha,
        seed=seed + 777,
        n_classes=10,
    )

    client_indices = all_client_indices[client_id]

    if len(client_indices) == 0:
        raise RuntimeError(
            f"O cliente {client_id} recebeu uma partição vazia."
        )

    client_train_set = Subset(
        full_train,
        client_indices,
    )

    generator = torch.Generator()
    generator.manual_seed(seed + 10001 + client_id)

    trainloader = DataLoader(
        client_train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
    )

    probe_loader = DataLoader(
        client_train_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )



    testloader = DataLoader(
        testset,
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
    )

    class_counts = np.bincount(
        targets[client_indices],
        minlength=10,
    )

    print(
        f"[Cliente {client_id}] "
        f"amostras={len(client_indices)}, "
        f"alpha={alpha}, "
        f"classes={class_counts.tolist()}",
        flush=True,
    )

    return trainloader, probe_loader, testloader

def probe_loss(
    net,
    probe_loader,
    max_batches: int = 5,
    device=DEVICE,
):
    net.to(device)
    net.eval()

    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch_index, (inputs, labels) in enumerate(probe_loader):
            if batch_index >= max_batches:
                break

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            loss = F.cross_entropy(
                outputs,
                labels,
                reduction="sum",
            )

            total_loss += float(loss.item())
            total_examples += int(labels.size(0))

    if total_examples == 0:
        raise RuntimeError(
            "Não foi possível calcular o probe_loss: "
            "o carregador não possui amostras."
        )

    result = total_loss / total_examples

    print(
        f"Probe loss antes do treinamento: "
        f"{result:.4f}",
        flush=True,
    )

    return result




def train(
    net,
    trainloader,
    steps=10,
    lr=0.01,
    device=DEVICE,
):
    net.to(device)
    net.train()

    optimizer = optim.SGD(
        net.parameters(),
        lr=lr,
        momentum=0.9,
    )

    data_iterator = iter(trainloader)

    total_loss = 0.0
    total_examples = 0

    for _ in range(steps):
        try:
            inputs, labels = next(data_iterator)
        except StopIteration:
            data_iterator = iter(trainloader)
            inputs, labels = next(data_iterator)

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = F.cross_entropy(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    mean_loss = total_loss / max(1, total_examples)

    print(
        f"Treinamento local finalizado: "
        f"steps={steps}, loss={mean_loss:.4f}"
    )

    return {"metric": mean_loss}

def test(net, testloader):
	net.eval()
	criterion = nn.CrossEntropyLoss()
	correct = 0
	total = 0
	with torch.no_grad():
		for inputs, labels in testloader:
			outputs = net(inputs)
			_, preds = torch.max(outputs, 1)
			correct += (preds == labels).sum().item()
			total += labels.size(0)
	accuracy = correct / total
	print(f'Test Accuracy: {accuracy:.4f}')
	return accuracy

def get_weights(net):
	return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
	params_dict = zip(net.state_dict().keys(), parameters)
	state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
	net.load_state_dict(state_dict, strict=True)
	return net