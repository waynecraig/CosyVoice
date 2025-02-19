FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/CosyVoice

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get update -y
RUN apt-get -y install build-essential

COPY cosyvoice /opt/CosyVoice/cosyvoice
COPY requirements.txt /opt/CosyVoice/requirements.txt
COPY server.py /opt/CosyVoice/server.py

RUN pip3 install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com