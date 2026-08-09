FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

WORKDIR /client

COPY FTDL ./FTDL
COPY client.py ./client.py
COPY task.py ./task.py
COPY requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip
RUN python -m pip install "flwr[simulation]==1.11.1"

RUN python -m pip install \
    torchvision==0.24.0 \
    torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu128

RUN python -m pip install ./FTDL/microservice_interconnect/.
RUN python -m pip install ./FTDL
RUN python -m pip install -r requirements.txt

CMD ["python", "-u", "client.py"]
