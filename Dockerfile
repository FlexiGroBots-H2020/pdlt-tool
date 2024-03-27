FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Metadata as described in the documentation
LABEL maintainer="mario.trivino@eviden.com" \
      version="1.0" \
      description="Image with CUDA 12.1.0, pytorch, and other dependencies."

# Arguments to build Docker Image using CUDA
ENV AM_I_DOCKER True
ENV BUILD_WITH_CUDA True

ENV CUDA_HOME /usr/local/cuda/
ENV DEBIAN_FRONTEND=noninteractive
USER root

# Install essential packages and Jupyter Notebook
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-dev \
    python3-opencv \
    x11-apps \
    wget \
    g++ && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch, torchvision, and torchaudio with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y git
    
WORKDIR /wd

COPY requirements.txt /wd
RUN pip install -r /wd/requirements.txt

RUN apt-get update && apt-get install ffmpeg python3-dev python3-setuptools libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev \
    libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk \
    libharfbuzz-dev libfribidi-dev libxcb1-dev -y
    

COPY DPT /wd/DPT/
COPY models /wd/models/
COPY README.md /wd
COPY smoothers /wd/smoothers/
COPY tools_clip /wd/tools_clip/
COPY deep_3d_calibration.py /wd
COPY deep_3d_realTime.py /wd
COPY deep_3d_utils.py /wd
COPY deep_sort /wd/deep_sort/
COPY yolov8 /wd/yolov8/
COPY trackers /wd/trackers/
COPY detectron2 /wd/detectron2/
COPY yolo_utils.py /wd
COPY distance_calibration_function.npy /wd
COPY mplot_thread.py /wd
COPY mplot2d.py /wd

COPY DepthAnything /wd/DepthAnything
WORKDIR /wd/DepthAnything
RUN pip install -r requirements.txt 

COPY PerspectiveFields /wd/PerspectiveFields/
WORKDIR /wd/PerspectiveFields
RUN pip install -r requirements.txt
RUN mim install mmcv 
RUN pip install -e .
WORKDIR /wd


RUN chmod -R 777 /wd


# Create a user with UID 1000 and GID 1000
RUN useradd -m -s /bin/bash -N -u 1000 jovyan


USER jovyan
#ENTRYPOINT ["python", "-u", "deep_3d_realTime.py"]
CMD ["tail", "-f", "/dev/null"]
