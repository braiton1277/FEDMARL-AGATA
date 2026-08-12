import os
import pickle
from collections import OrderedDict

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
