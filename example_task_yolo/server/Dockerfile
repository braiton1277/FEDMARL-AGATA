FROM ultralytics/ultralytics:latest
WORKDIR /server

COPY FTDL ./FTDL
COPY server.py ./server.py
COPY task.py ./task.py

RUN pip install --upgrade pip
RUN pip install flwr[simulation]==1.11.1
RUN pip install ./FTDL/microservice_interconnect/.
RUN pip install ./FTDL

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

CMD ["python3","-u","server.py"]