FROM python:3.9-slim-buster

COPY docker/requirements.txt .

RUN apt-get update && apt-get -y upgrade \
  && apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo "Running $(conda --version)" && \
    conda init bash  \
    && . /root/.bashrc  \
    && conda update conda  \
    && conda create -n python-app  \
    && conda activate python-app  \
    && conda install python=3.9 pip  \
    && pip install -r requirements.txt \
    && echo 'print("Hello World!")' > SvmScript.py
RUN echo 'conda activate python-app \n\
alias python-app="python python-app.py"' >> /root/.bashrc
ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
CMD ["python SvmScript.py"]