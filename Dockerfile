# For more information, please refer to https://aka.ms/vscode-docker-python
FROM public.ecr.aws/j1r0q0g6/notebooks/notebook-servers/jupyter-pytorch-cuda-full:v1.5.0

USER root

RUN apt-get update && apt-get install -y python3-opencv wget g++

WORKDIR /wd

RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
COPY requirements.txt /wd
RUN pip install -r requirements.txt

COPY Detic/ /wd/Detic/
COPY DPT /wd/DPT/
COPY models /wd/models/
COPY README.md /wd
COPY smoothers /wd/smoothers/
COPY tools_clip /wd/tools_clip/
COPY deep_3d_calibration.py /wd
COPY deep_3d_main.py /wd
COPY deep_3d_utils.py /wd

USER jovyan
ENTRYPOINT ["python", "-u", "deep_3d_main.py"]
