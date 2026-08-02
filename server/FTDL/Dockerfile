FROM python:3.12

COPY install-docker.sh /usr/local
RUN bash /usr/local/install-docker.sh

COPY test /usr/local/test
WORKDIR /usr/local/test

COPY ftdl ./FTDL/ftdl
COPY pyproject.toml ./FTDL/pyproject.toml
COPY setup.py ./FTDL/setup.py
COPY microservice_interconnect ./FTDL/microservice_interconnect

RUN pip install --upgrade pip
RUN pip install ./FTDL/microservice_interconnect
RUN pip install ./FTDL

LABEL name="test_ftdl"

CMD ["python3","-u","/usr/local/test/main.py"]
