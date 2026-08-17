import os
import pickle
from collections import OrderedDict



from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(base_weights_path):
    num_classes = int(os.environ.get("NUM_CLASSES", "1"))

    model = YOLO(base_weights_path)

    custom_model = DetectionModel(
        model.model.yaml,
        ch=3,
        nc=num_classes,
        verbose=False,
    )

    custom_model.load(model.model)
    custom_model.args = get_cfg()
    custom_model.names = {i: str(i) for i in range(num_classes)}

    model.model = custom_model
    model.model.to(DEVICE)

    return model

class Net:
    def __init__(self, base_weights_path=None) -> None:
        self.base_weights_path = base_weights_path or os.environ.get(
            "BASE_WEIGHTS_PATH", "yolo11n.pt"
        )
        self.model = None
        self.device = DEVICE

    def to(self, device: torch.device):
        self.device = device
        if self.model is not None:
            self.model.model.to(device)
        return self

def get_weights(net: Net) -> list[np.ndarray]:
    if net.model is None:
        net.model = build_model(net.base_weights_path)
        net.model.model.to(net.device)

    return [tensor.cpu().numpy() for tensor in net.model.model.state_dict().values()]

def set_weights(net: Net, parameters: list[np.ndarray]):
    if net.model is None:
        net.model = YOLO(net.base_weights_path)
        net.model.model.to(net.device)

    keys = list(net.model.model.state_dict().keys())
    updated_weights = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    state_dict = OrderedDict(updated_weights)
    net.model.model.load_state_dict(state_dict, strict=False)
    return net

def net_to_bytes(net: Net)->bytes:
    params = get_weights(net) # make sure the model is loaded and weights are on the CPU
    return pickle.dumps(params)

def bytes_to_net(net: Net, bytes_data: bytes)->Net:
    data_to_put_on_the_net = pickle.loads(bytes_data)
    return set_weights(net, data_to_put_on_the_net)


def load_probe_loader(
    data_yaml="/data/cityscapes/data.yaml",
    batch_size=2,
):
    data = check_det_dataset(
        data_yaml,
        autodownload=False,
    )

    cfg = get_cfg(
        overrides={
            "task": "detect",
            "imgsz": 640,
            "batch": batch_size,
            "workers": 0,
            "rect": True,
            "cache": False,
        }
    )

    dataset = build_yolo_dataset(
        cfg=cfg,
        img_path=data["val"],
        batch=batch_size,
        data=data,
        mode="val",
        rect=True,
        stride=32,
    )

    loader = build_dataloader(
        dataset=dataset,
        batch=batch_size,
        workers=0,
        shuffle=False,
        rank=-1,
        device=DEVICE,
    )

    return loader


def evaluate_server_loss(
    net,
    validation_loader,
    max_batches=20,
):
    model = net.model.model
    was_training = model.training

    buffer_backup = {
        name: buffer.detach().clone()
        for name, buffer in model.named_buffers()
    }

    model.train()

    total_loss = 0.0
    total_examples = 0

    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(validation_loader):
                if batch_index >= max_batches:
                    break

                batch = {
                    key: (
                        value.to(DEVICE, non_blocking=True)
                        if torch.is_tensor(value)
                        else value
                    )
                    for key, value in batch.items()
                }

                batch["img"] = batch["img"].float() / 255.0

                loss, _ = model.loss(batch)

                batch_size = int(
                    batch["img"].shape[0]
                )

                total_loss += float(
                    loss.detach().sum().item()
                )

                total_examples += batch_size

        if total_examples == 0:
            raise RuntimeError(
                "Nenhuma imagem usada na validação."
            )

        validation_loss = (
            total_loss / float(total_examples)
        )

        if not np.isfinite(validation_loss):
            raise RuntimeError(
                "Loss de validação não é finita."
            )

        return float(validation_loss)

    finally:
        with torch.no_grad():
            current_buffers = dict(model.named_buffers())

            for name, saved_buffer in buffer_backup.items():
                current_buffers[name].copy_(saved_buffer)

        model.train(was_training)








def server_reference_grad(
    net,
    probe_loader,
    batches=1,
):
    model = net.model.model

    was_training = model.training


    buffer_backup = {
        name: buffer.detach().clone()
        for name, buffer in model.named_buffers()
    }

    model.train()
    model.zero_grad(set_to_none=True)

    used_batches = 0

    try:
        for batch_index, batch in enumerate(probe_loader):
            if batch_index >= batches:
                break


            batch = {
                key: (
                    value.to(DEVICE, non_blocking=True)
                    if torch.is_tensor(value)
                    else value
                )
                for key, value in batch.items()
            }


            batch["img"] = batch["img"].float() / 255.0

            loss, _ = model.loss(batch)

            total_loss = loss.sum()

            total_loss.backward()

            used_batches += 1

        if used_batches == 0:
            raise RuntimeError(
                "Nenhum batch disponível para o probe."
            )

        gradients = []

        for parameter in model.parameters():
            if parameter.grad is None:
                gradients.append(
                    torch.zeros_like(parameter).reshape(-1)
                )
            else:
                gradients.append(
                    parameter.grad.detach().reshape(-1)
                )

        reference_gradient = torch.cat(
            gradients
        )

        reference_gradient /= float(used_batches)

    finally:
        model.zero_grad(set_to_none=True)

        with torch.no_grad():
            current_buffers = dict(
                model.named_buffers()
            )

            for name, saved_buffer in buffer_backup.items():
                current_buffers[name].copy_(
                    saved_buffer
                )

        model.train(was_training)

    return reference_gradient.cpu().numpy()

