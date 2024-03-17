FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

ENV PYTHON_VERSION 3.11.4
ENV PYENV_ROOT /root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential \
        libffi-dev \
        libssl-dev \
        zlib1g-dev \
        liblzma-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        llvm \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev \
        libgl1-mesa-dev

RUN curl https://pyenv.run | bash

RUN pyenv install $PYTHON_VERSION
RUN pyenv global $PYTHON_VERSION

RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN pip install wheel==0.43.0

RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
RUN pip install --user -e detectron2_repo --no-build-isolation

COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN pip install git+https://github.com/cocodataset/panopticapi.git
RUN pip install hydra-core --upgrade
RUN pip install einops --upgrade 

ENV PYTHONPATH "/workspace/maskdiffusion:/workspace/maskdiffusion/scripts:/workspace/maskdiffusion/third_party:/workspace/maskdiffusion/third_party/Mask2Former"
