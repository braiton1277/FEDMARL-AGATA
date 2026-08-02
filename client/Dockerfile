#FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime
FROM python:3.12
WORKDIR /client

COPY FTDL ./FTDL
COPY client.py ./client.py
COPY task.py task.py

RUN pip install --upgrade pip
RUN pip install flwr[simulation]==1.11.1
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install ./FTDL/microservice_interconnect/.
RUN pip install ./FTDL
#RUN pip uninstall opencv-python
#RUN pip install opencv-python-headless

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

CMD ["python3","-u","client.py"]
