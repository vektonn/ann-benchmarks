FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y python3-pip python3-numpy python3-scipy build-essential git curl wget

WORKDIR /home/app

COPY requirements.txt run_algorithm.py ./

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3", "-u", "run_algorithm.py"]
