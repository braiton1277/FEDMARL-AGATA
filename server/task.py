
import pickle
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

SERVER_DEVICE = torch.device("cpu")


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
    dataset,
    per_class: int = 200,
    n_classes: int = 10,
    seed: int = 2048 + 4242,
) -> List[int]:
    rng = np.random.RandomState(seed)

    label_to_indices: Dict[int, List[int]] = {
        class_id: []
        for class_id in range(n_classes)
    }

    for index in range(len(dataset)):
        _, label = dataset[index]

        label_to_indices[
            int(label)
        ].append(index)

    validation_indices = []

    for class_id in range(n_classes):
        class_indices = label_to_indices[class_id]

        rng.shuffle(class_indices)

        validation_indices.extend(
            class_indices[:per_class]
        )

    rng.shuffle(validation_indices)

    return validation_indices



def get_weights(net):
	return [val.cpu().numpy() for _, val in net.state_dict().items()]
 
def set_weights(net, parameters):
	params_dict = zip(net.state_dict().keys(), parameters)
	state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
	net.load_state_dict(state_dict, strict=True)
	return net
    
def net_to_bytes(net: Net)->bytes:
    params = get_weights(net) # make sure the model is loaded and weights are on the CPU
    return pickle.dumps(params)

def bytes_to_net(net: Net, bytes_data: bytes)->Net:
    data_to_put_on_the_net = pickle.loads(bytes_data)
    return set_weights(net, data_to_put_on_the_net)


def load_server_validation(
    data_path: str,
    seed: int = 2048,
    per_class: int = 200,
    batch_size: int = 256,
):
    mean = (
        0.4914,
        0.4822,
        0.4465,
    )

    std = (
        0.2470,
        0.2435,
        0.2616,
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean,
                std,
            ),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=transform,
    )

    validation_indices = (
        make_server_val_balanced(
            dataset=train_dataset,
            per_class=per_class,
            n_classes=10,
            seed=seed + 4242,
        )
    )

    validation_set = Subset(
        train_dataset,
        validation_indices,
    )

    validation_loader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(
        f"Server validation: "
        f"{len(validation_set)} amostras, "
        f"{per_class} por classe.",
        flush=True,
    )

    return validation_loader



def flatten_grads(model):
    gradients = []

    for parameter in model.parameters():
        if parameter.grad is None:
            gradient = torch.zeros_like(
                parameter
            )
        else:
            gradient = parameter.grad

        gradients.append(
            gradient.detach().reshape(-1)
        )

    return torch.cat(
        gradients
    )


def server_reference_grad(
    model,
    validation_loader,
    batches: int = 10,
    device=SERVER_DEVICE,
):
    model.to(device)
    model.train()

    model.zero_grad(
        set_to_none=True
    )

    processed_batches = 0

    for batch_index, (
        inputs,
        labels,
    ) in enumerate(validation_loader):
        if batch_index >= batches:
            break

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        loss = F.cross_entropy(
            outputs,
            labels,
        )

        loss.backward()

        processed_batches += 1

    if processed_batches == 0:
        raise RuntimeError(
            "Nenhum batch foi usado para calcular "
            "o gradiente de referência."
        )

    reference_gradient = (
        flatten_grads(model)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    model.zero_grad(
        set_to_none=True
    )

    return reference_gradient