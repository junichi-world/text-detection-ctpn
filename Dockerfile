ARG BASE_IMAGE=nvcr.io/nvidia/tensorflow:25.01-tf2-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/text-detection-ctpn

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir "numpy<2" && \
    python -m pip install --no-cache-dir \
      easydict \
      opencv-python \
      Pillow \
      matplotlib \
      tqdm \
      PyYAML \
      scipy \
      pandas \
      Cython \
      setuptools

COPY . /workspace/text-detection-ctpn

RUN chmod +x scripts/*.sh lib/utils/make.sh && \
    /workspace/text-detection-ctpn/scripts/setup_env.sh --skip-pip

CMD ["/bin/bash"]
