ARG BASE_IMAGE
FROM $BASE_IMAGE

# Make data directory
RUN mkdir -p /data

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y install \
    build-essential \
    cmake \
    git \
    python3-dev \
    python3-setuptools \
    python3-pip \
    nano \
    vim \
    zsh

# Install tensorboardX and pandas (and PyCharm development tools for debugging) but delete cache afterwards
RUN pip install tensorboardX pandas && \
    pip install pydevd-pycharm~=211.7628.24 && \
    rm -rf /root/.cache/pip

# install oh-my-zsh
RUN git clone "https://github.com/robbyrussell/oh-my-zsh.git" "${HOME}/.oh-my-zsh" && \
    cp "${HOME}/.oh-my-zsh/templates/zshrc.zsh-template" "${HOME}/.zshrc"
