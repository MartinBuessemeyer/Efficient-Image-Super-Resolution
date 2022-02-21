# Make data directory
RUN mkdir -p /data

export DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get -y install \
    build-essential \
    cmake \
    git \
    graphviz \
    python3-dev \
    python3-setuptools \
    python3-pip \
    vim \
    zsh

# Install own dependencies
RUN pip install imageio \
    matplotlib \
    numpy \
    opencv-python-headless \
    pillow \
    scikit-image \
    torchvision \
    tqdm \
    wandb \
    pandas

# Install latest pytorch
pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install development tool for PyCharm for remote debugging
RUN pip install pydevd-pycharm~=211.7628.24

# Save some space
RUN rm -rf /root/.cache/pip

# Install oh-my-zsh
RUN git clone "https://github.com/robbyrussell/oh-my-zsh.git" "${HOME}/.oh-my-zsh"
RUN cp "${HOME}/.oh-my-zsh/templates/zshrc.zsh-template" "${HOME}/.zshrc"
