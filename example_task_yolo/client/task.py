import os
import tempfile
from collections import OrderedDict

import numpy as np
import torch
import yaml
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
    custom_model.names = {i: str(i) for i in range(num_classes)}

    model.model = custom_model
    model.model.to(DEVICE)

    return model

class Net:
    def __init__(self, base_weights_path=None, data_yaml_path=None) -> None:
        self.base_weights_path = base_weights_path or os.environ.get(
            "BASE_WEIGHTS_PATH", "/data/yolo11n.pt"
        )
        self.data_yaml_path = data_yaml_path or os.environ.get(
            "DATA_YAML_PATH", "/data/yolo_data.yaml"
        )
        self.model = None
        self.device = DEVICE

    def set_model(self, model:YOLO):
        self.model = model

    def train(self, batch_size, epochs):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_results = self.model.train(
                data=self.data_yaml_path,
                epochs=epochs,
                batch=batch_size,
                plots=False,
                verbose=False,
                workers=0,
                project=tmpdir,
            )
        return train_results

    def to(self, device: torch.device):
        self.device = device
        if self.model is not None:
            self.model.model.to(device)
        return self


# ------------------------------
# Helper functions
# ------------------------------

def _resolve_split_path(cfg: dict, data_yaml_path: str, key: str):
    yaml_dir = os.path.dirname(os.path.abspath(data_yaml_path))

    base = cfg.get("path", "")
    if base:
        if not os.path.isabs(base):
            base = os.path.join(yaml_dir, base)
    else:
        base = yaml_dir

    split = cfg.get(key)
    if split is None:
        return None

    if isinstance(split, list):
        resolved = []
        for p in split:
            resolved.append(p if os.path.isabs(p) else os.path.join(base, p))
        return resolved

    return split if os.path.isabs(split) else os.path.join(base, split)


def _count_images_from_path(path):
    if path is None:
        return 0

    if isinstance(path, list):
        return sum(_count_images_from_path(p) for p in path)

    if os.path.isfile(path) and path.lower().endswith(".txt"):
        with open(path, "r") as f:
            return sum(1 for line in f if line.strip())

    if os.path.isdir(path):
        count = 0
        for root, _, files in os.walk(path):
            for fn in files:
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    count += 1
        return count

    return 0


def _get_data_yaml_path(data_path: str) -> str:
    env_path = os.environ.get("DATA_YAML_PATH")
    if env_path:
        return env_path
    name = os.environ.get("DATA_YAML_NAME", "yolo_data.yaml")
    return os.path.join(data_path, name)


class ProxyDataset:
    def __init__(self, data_yaml: str, split: str) -> None:
        self.data_yaml = data_yaml
        self.split = split
        self._size = self._count_images()

    def _count_images(self) -> int:
        with open(self.data_yaml, "r") as f:
            cfg = yaml.safe_load(f)

        if self.split == "train":
            key = "train"
        else:
            key = "val" if "val" in cfg else "test"

        split_path = _resolve_split_path(cfg, self.data_yaml, key)
        return _count_images_from_path(split_path)

    def __len__(self) -> int:
        return self._size


class ProxyLoader:
    def __init__(self, dataset: ProxyDataset) -> None:
        self.dataset = dataset


# ------------------------------
# AGATA-required API
# ------------------------------

def load_data(data_path: str):
    data_yaml = _get_data_yaml_path(data_path)
    train_set = ProxyDataset(data_yaml=data_yaml, split="train")
    test_set = ProxyDataset(data_yaml=data_yaml, split="val")
    return ProxyLoader(train_set), ProxyLoader(test_set)


def load_parameters(model_path: str) -> YOLO:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return YOLO(model_path).to(device)


def save_parameters(model: YOLO, out_path: str):
    model.ckpt["model"] = model.model
    if hasattr(model, "ema"):
        del model.ckpt["ema"]
    model.save(out_path)
    model.model.to(DEVICE)
    print(f"Model weights saved at {out_path}")


def train(net: Net, trainloader, valloader, epochs: int, device: torch.device):
    net.to(device)

    ckpt_path = os.environ.get("LOCAL_CKPT_PATH", "/data/yolo_local_client.pt")

    if net.model is None:
        net.model = build_model(net.base_weights_path)

    net.model.model.to(device)

    batch_size = int(os.environ.get("BATCH_SIZE", "8"))

    with tempfile.TemporaryDirectory() as tmpdir:
        train_results = net.model.train(
            data=net.data_yaml_path,
            epochs=epochs,
            batch=batch_size,
            plots=False,
            verbose=False,
            workers=0,
            project=tmpdir,
        )

    save_parameters(net.model, ckpt_path)

    metrics = {}
    try:
        box_metrics = getattr(train_results, "box", None)
        if box_metrics is not None and hasattr(box_metrics, "map50"):
            metrics["map50"] = float(box_metrics.map50)
    except Exception as e:
        print(f"[train] Fail to extract metrics: {e}")

    if "map50" not in metrics:
        metrics["map50"] = 0.0

    return {"metric": metrics["map50"]}


def test(net: Net, testloader):
    net.to(DEVICE)

    if net.model is None:
        net.model = build_model(net.base_weights_path)
        net.model.model.to(DEVICE)

    results = net.model.val(
        data=net.data_yaml_path,
        batch=int(os.environ.get("BATCH_SIZE", "8")),
        workers=0,
        verbose=False,
        plots=False,
    )

    metrics = {}
    try:
        box_metrics = getattr(results, "box", None)
        if box_metrics is not None and hasattr(box_metrics, "map50"):
            metrics["map50"] = float(box_metrics.map50)
    except Exception as e:
        print(f"[test] Fail to extract metrics: {e}")

    if "map50" not in metrics:
        metrics["map50"] = 0.0

    return metrics["map50"]


def get_weights(net: Net) -> list[np.ndarray]:
    if net.model is None:
        net.model = build_model(net.base_weights_path)
        net.model.model.to(net.device)

    return [tensor.cpu().numpy() for tensor in net.model.model.state_dict().values()]


def set_weights(net: Net, parameters: list[np.ndarray]):
    if net.model is None:
        net.model = build_model(net.base_weights_path)
        net.model.model.to(net.device)

    keys = list(net.model.model.state_dict().keys())
    updated_weights = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    state_dict = OrderedDict(updated_weights)
    net.model.model.load_state_dict(state_dict, strict=False)
    return net