
# Make data directory
RUN mkdir -p /data

export DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get -y install \
    build-essential \
    cmake \
    git \
    python3-dev \
    python3-setuptools \
    python3-pip \
    nano \
    vim \
    zsh

# install own dependencies
RUN pip install torchvision pillow opencv-python-headless numpy scikit-image imageio matplotlib tqdm torchviz

# Install tensorboardX and pandas
RUN pip install tensorboardX pandas wandb

# Install development tool for PyCharm for remote debugging
RUN pip install pydevd-pycharm~=211.7628.24

# save some space
RUN rm -rf /root/.cache/pip

# install oh-my-zsh
RUN git clone "https://github.com/robbyrussell/oh-my-zsh.git" "${HOME}/.oh-my-zsh"
RUN cp "${HOME}/.oh-my-zsh/templates/zshrc.zsh-template" "${HOME}/.zshrc"
