FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04
RUN mkdir /app
RUN mkdir /app/models
RUN mkdir /app/src

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys A4B469963BF863CC
RUN apt update
RUN apt install -y nvidia-utils-470 nvidia-driver-470
RUN apt install -y openjdk-11-jdk
RUN apt install -y python3 python3-pip
RUN apt install -y libopenmpi-dev
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
RUN pip install --upgrade requests

COPY ./src/ /app/src/

RUN torch-model-archiver --model-name "wafl-llm" --version 0.0.1 \
                         --handler /app/src/handler.py --export-path /app/models/

COPY config.properties /app/

CMD ["torchserve", "--start", "--model-store", "models", "--models", "bot=wafl-llm.mar", "--foreground"]
