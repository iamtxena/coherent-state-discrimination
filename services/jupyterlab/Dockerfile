FROM ubuntu:20.04 as ubuntu-miniconda-cuda

# Leave these args here to better use the Docker build cache
ARG CONDA_VERSION=py39_4.10.3
ARG SHA256SUM=1ea2f885b4dbc3098662845560bc64271eb17085387a70c2ba3f29fff6f8d52f

RUN apt update
RUN apt-get -y upgrade
RUN apt install -y bash git curl

RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh --output miniconda.sh && \
    bash miniconda.sh -b && \
    echo "${SHA256SUM}  miniconda.sh" > miniconda.sha256 && \
    if ! sha256sum -c miniconda.sha256; then exit 1; fi && \
    mkdir -p /opt && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh miniconda.sha256 && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete

RUN /opt/conda/bin/conda clean -afy

ENV PATH /bin:/sbin:/usr/bin:$PATH
ENV PATH /opt/conda/bin:$PATH
RUN conda init
RUN conda install scipy numpy matplotlib sympy

RUN pip install --upgrade pip

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Madrid
RUN apt-get install -y tzdata

RUN apt install -y pciutils software-properties-common
WORKDIR /root
RUN add-apt-repository ppa:graphics-drivers/ppa && \
    apt install -y nvidia-driver-495

RUN apt install wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/11.5.1/local_installers/cuda-repo-ubuntu2004-11-5-local_11.5.1-495.29.05-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2004-11-5-local_11.5.1-495.29.05-1_amd64.deb && \
    apt-key add /var/cuda-repo-ubuntu2004-11-5-local/7fa2af80.pub && \
    apt-get update && \
    apt-get -y install cuda
RUN export PATH=/usr/local/cuda-11.5/bin${PATH:+:${PATH}} && \
    export LD_LIBRARY_PATH=/usr/local/cuda-11.5/lib64\
    ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

FROM ubuntu-miniconda-cuda as debian-miniconda-jupyterlab

WORKDIR /root/.jupyter


RUN apt update
RUN apt install -y gcc musl-dev python3-dev

RUN pip install --upgrade pip

RUN mkdir /anaconda && cd /anaconda
RUN pip install --upgrade pip
RUN pip install jupyterlab && \
    jupyter notebook --generate-config

WORKDIR /root/.jupyter
COPY jupyter_notebook_config.py jupyter_notebook_config.py

ARG CUSTOM_GITHUB_USER
ARG CUSTOM_GITHUB_TOKEN

RUN cd /root && \
    git clone https://${CUSTOM_GITHUB_USER}:${CUSTOM_GITHUB_TOKEN}@github.com/iamtxena/coherent-state-discrimination.git && \
    cd coherent-state-discrimination && \
    git checkout using_ddbb && \
    cd services/library && \
    pip install .

WORKDIR /anaconda
ENV PATH /bin:/sbin:/usr/bin:$PATH
ENV PATH /opt/conda/bin:$PATH
ENV SHELL=/bin/bash
# Add conda env hook
COPY ./conda-activate.sh /usr/local/bin/before-notebook.d/

CMD ["jupyter","lab", "--no-browser"]