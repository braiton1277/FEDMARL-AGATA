FROM ultralytics/ultralytics:latest
WORKDIR /client

COPY FTDL ./FTDL
COPY client.py ./client.py
COPY task.py ./task.py

RUN pip install flwr[simulation]==1.11.1
RUN pip install ./FTDL/microservice_interconnect/.
RUN pip install ./FTDL

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

CMD ["python3","-u","client.py"]