# For more information, please refer to https://aka.ms/vscode-docker-python
FROM public.ecr.aws/j1r0q0g6/notebooks/notebook-servers/jupyter-pytorch-cuda-full:v1.5.0

USER root

RUN apt-get update && apt-get install -y python3-opencv wget g++


WORKDIR /wd

RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
COPY requirements.txt /wd
RUN pip install -r requirements.txt

RUN apt-get update && apt-get install ffmpeg python3-dev python3-setuptools libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev \
    libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk \
    libharfbuzz-dev libfribidi-dev libxcb1-dev -y

COPY Detic/ /wd/Detic/
COPY DPT /wd/DPT/
COPY models /wd/models/
COPY README.md /wd
COPY smoothers /wd/smoothers/
COPY tools_clip /wd/tools_clip/
COPY deep_3d_calibration.py /wd
COPY deep_3d_main.py /wd
COPY deep_3d_realTime.py /wd
COPY deep_3d_utils.py /wd
COPY deep_sort /wd/deep_sort/
COPY yolov8 /wd/yolov8/
COPY trackers /wd/trackers/
COPY clip /wd/clip/
COPY detectron2 /wd/detectron2/
COPY yolo_utils.py /wd
COPY distance_calibration_function.npy /wd


RUN chmod -R 777 /wd

USER jovyan
ENTRYPOINT ["python", "-u", "deep_3d_realTime.py"]
