FROM python:3.9 as ubuntu-miniconda

# Leave these args here to better use the Docker build cache
ARG CONDA_VERSION=py39_4.10.3
ARG SHA256SUM=1ea2f885b4dbc3098662845560bc64271eb17085387a70c2ba3f29fff6f8d52f

RUN apt update
RUN apt install bash
RUN apt install git
RUN apt install curl
RUN curl -o rsvisa.deb https://scdn.rohde-schwarz.com/ur/pws/dl_downloads/dl_application/application_notes/1dc02___rs_v/rsvisa_5.12.3_amd64.deb
RUN apt install -f -y ./rsvisa.deb
#RUN apt install rsvisa

# hadolint ignore=DL3018

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


FROM ubuntu-miniconda as ubuntu-miniconda-jupyterlab

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
WORKDIR /anaconda
RUN cd /anaconda && \
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