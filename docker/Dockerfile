FROM nvcr.io/nvidia/cuda:11.7.0-base-ubuntu20.04 AS sd-dino

USER root
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update 
RUN apt-get -y install pybind11-dev \
    git \
    wget \
    python3-tk \
    python3-pip 

RUN pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html


RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
RUN git clone https://github.com/facebookresearch/detectron2.git
RUN pip install pycocotools
RUN pip install cython

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1-mesa-glx \
      libglib2.0-0 \
      libxrender1 \
      libxext6 \
      libx11-6 \
  && rm -rf /var/lib/apt/lists/*
  
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      wget \
      python3-tk \
      python3-pip \
      pybind11-dev \
      # hier die OpenGL-Libs:
      libgl1-mesa-glx \
      libglib2.0-0 \
      libxrender1 \
      libxext6 \
      libx11-6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -e detectron2
COPY . /sd-dino


WORKDIR /sd-dino

# Install dependencies one by one, skipping failed installs
RUN pip install -r requirements.txt

RUN ln -s /usr/bin/python3 /usr/bin/python

CMD ["/bin/bash"]


