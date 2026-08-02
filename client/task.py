from collections import OrderedDict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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


class SwitchableTargetedLabelFlipSubset(Dataset):
    """
    Subconjunto do CIFAR-10 cujo ataque pode ser ligado/desligado por rodada.

    O sorteio que define quais exemplos serão invertidos é feito uma única vez,
    de maneira determinística. Assim, para uma mesma seed e taxa de ataque,
    os mesmos exemplos são alterados em todas as rodadas.
    """

    DEFAULT_TARGET_MAP = {
        0: 8,
        8: 0,
        1: 9,
        9: 1,
        2: 6,
        6: 2,
        3: 5,
        5: 3,
        4: 7,
        7: 4,
    }

    def __init__(
        self,
        base_dataset,
        indices: List[int],
        seed: int,
        enabled: bool = False,
        attack_rate: float = 0.0,
        target_map: Dict[int, int] | None = None,
    ):
        self.base_dataset = base_dataset
        self.indices = [int(index) for index in indices]
        self.enabled = bool(enabled)
        self.attack_rate = self._validate_rate(attack_rate)

        mapping = (
            self.DEFAULT_TARGET_MAP
            if target_map is None
            else target_map
        )
        self.target_map = {
            int(source): int(target)
            for source, target in mapping.items()
        }

        rng = np.random.RandomState(int(seed))
        self.attack_draw = rng.rand(len(self.indices)).astype(np.float32)

        base_targets = np.asarray(
            self.base_dataset.targets,
            dtype=np.int64,
        )
        self.original_labels = base_targets[self.indices].copy()
        self.flipped_labels = np.asarray(
            [
                self.target_map.get(int(label), int(label))
                for label in self.original_labels
            ],
            dtype=np.int64,
        )

        unchanged = self.flipped_labels == self.original_labels
        if bool(np.any(unchanged)):
            raise ValueError(
                "O mapa de label flipping deixou ao menos um rótulo "
                "inalterado."
            )

    @staticmethod
    def _validate_rate(rate: float) -> float:
        rate = float(rate)
        if not 0.0 <= rate <= 1.0:
            raise ValueError(
                f"attack_rate deve estar entre 0 e 1, recebido: {rate}"
            )
        return rate

    def set_attack(self, enabled: bool, rate: float) -> None:
        self.enabled = bool(enabled)
        self.attack_rate = self._validate_rate(rate)

    def attack_status(self) -> Dict[str, int | float]:
        active_flips = 0
        if self.enabled and self.attack_rate > 0.0:
            active_flips = int(
                np.count_nonzero(
                    self.attack_draw < self.attack_rate
                )
            )

        return {
            "enabled": int(self.enabled),
            "attack_rate": float(self.attack_rate),
            "active_flips": int(active_flips),
            "num_examples": int(len(self.indices)),
        }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        base_index = self.indices[position]
        image, clean_label = self.base_dataset[base_index]

        label = int(clean_label)

        should_flip = (
            self.enabled
            and self.attack_rate > 0.0
            and self.attack_draw[position] < self.attack_rate
        )

        if should_flip:
            label = int(self.flipped_labels[position])

        return image, label


def set_label_attack(
    loader_or_dataset,
    enabled: bool,
    rate: float,
) -> Dict[str, int | float]:
    """
    Configura o ataque no dataset associado a um DataLoader.
    """
    dataset = getattr(
        loader_or_dataset,
        "dataset",
        loader_or_dataset,
    )

    if not isinstance(
        dataset,
        SwitchableTargetedLabelFlipSubset,
    ):
        raise TypeError(
            "O dataset não suporta ativação dinâmica de label flipping."
        )

    dataset.set_attack(
        enabled=enabled,
        rate=rate,
    )

    return dataset.attack_status()


def make_server_val_balanced(
    targets,
    per_class: int = 200,
    n_classes: int = 10,
    seed: int = 0,
) -> List[int]:
    """
    Reserva uma quantidade balanceada de imagens de cada classe
    para a validação do servidor.
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
        validation_indices.extend(
            class_indices[:per_class]
        )

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
        raise ValueError(
            "n_clients deve ser maior que zero"
        )

    if alpha <= 0:
        raise ValueError(
            "alpha deve ser maior que zero"
        )

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

        difference = len(class_indices) - counts.sum()

        if difference > 0:
            selected_clients = rng.choice(
                n_clients,
                size=difference,
                replace=True,
            )

            for selected_client in selected_clients:
                counts[selected_client] += 1

        elif difference < 0:
            available_clients = np.where(
                counts > 0
            )[0]

            selected_clients = rng.choice(
                available_clients,
                size=-difference,
                replace=True,
            )

            for selected_client in selected_clients:
                counts[selected_client] -= 1

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

    # Mesma normalização utilizada pelo servidor e pelo FEDMARL original.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
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

    client_dataset = SwitchableTargetedLabelFlipSubset(
        base_dataset=full_train,
        indices=client_indices,
        seed=seed + 1000 + client_id,
        enabled=False,
        attack_rate=0.0,
    )

    generator = torch.Generator()
    generator.manual_seed(
        seed + 10001 + client_id
    )

    trainloader = DataLoader(
        client_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
    )

    # O probe compartilha o mesmo dataset com o treino.
    # Assim, quando o ataque é ativado, ambos veem os rótulos invertidos.
    probe_loader = DataLoader(
        client_dataset,
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
        f"classes_limpas={class_counts.tolist()}",
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
        for batch_index, (
            inputs,
            labels,
        ) in enumerate(probe_loader):
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
        loss = F.cross_entropy(
            outputs,
            labels,
        )

        loss.backward()
        optimizer.step()

        current_batch_size = labels.size(0)
        total_loss += (
            loss.item() * current_batch_size
        )
        total_examples += current_batch_size

    mean_loss = total_loss / max(
        1,
        total_examples,
    )

    print(
        f"Treinamento local finalizado: "
        f"steps={steps}, loss={mean_loss:.4f}",
        flush=True,
    )

    return {
        "metric": float(mean_loss),
    }


def test(
    net,
    testloader,
    device=DEVICE,
):
    net.to(device)
    net.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()
            total += labels.size(0)

    accuracy = correct / max(1, total)

    print(
        f"Test Accuracy: {accuracy:.4f}",
        flush=True,
    )

    return accuracy


def get_weights(net):
    return [
        value.cpu().numpy()
        for _, value in net.state_dict().items()
    ]


def set_weights(net, parameters):
    params_dict = zip(
        net.state_dict().keys(),
        parameters,
    )

    state_dict = OrderedDict({
        key: torch.tensor(value)
        for key, value in params_dict
    })

    net.load_state_dict(
        state_dict,
        strict=True,
    )

    return net