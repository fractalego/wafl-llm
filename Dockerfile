FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
RUN mkdir /app
RUN mkdir /app/models
RUN mkdir /app/wafl_llm

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
RUN apt install -y python3-packaging
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
RUN pip install --upgrade requests

COPY ./wafl_llm/ /app/wafl_llm/
COPY config.json /app/

RUN torch-model-archiver --model-name "llm" --version 0.0.1 \
                         --handler /app/wafl_llm/llm_handler.py --extra-files /app/wafl_llm/config.json --export-path /app/models/

RUN torch-model-archiver --model-name "speaker" --version 0.0.1 \
                         --handler /app/wafl_llm/speaker_handler.py --extra-files /app/wafl_llm/config.json --export-path /app/models/

RUN torch-model-archiver --model-name "whisper" --version 0.0.1 \
                         --handler /app/wafl_llm/whisper_handler.py --extra-files /app/wafl_llm/config.json --export-path /app/models/

RUN torch-model-archiver --model-name "sentence_embedder" --version 0.0.1 \
                         --handler /app/wafl_llm/sentence_embedder_handler.py --extra-files /app/wafl_llm/config.json --export-path /app/models/ \

RUN torch-model-archiver --model-name "entailer" --version 0.0.1 \
                     --handler /app/wafl_llm/entailer_handler.py --extra-files /app/wafl_llm/config.json --export-path /app/models/

COPY config.properties /app/
CMD ["torchserve", "--start", "--model-store", "models", \
     "--models", \
     "bot=llm.mar", \
     "speaker=speaker.mar", \
     "whisper=whisper.mar", \
     "sentence_embedder=sentence_embedder.mar", \
     "--foreground"]
