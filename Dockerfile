FROM python:3.9-buster

WORKDIR /var

ADD . /var

RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip &&  pip install --no-cache-dir -r requirements.txt

ENV NAME BachelorThesis
ENV PYTHONPATH /var

RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc libsndfile1

RUN mkdir "dataset"

CMD ["python", "SvmScript.py"]


FROM continuumio/anaconda3
MAINTAINER "Frederick Rauh"

RUN apt-get update && apt-get install -y libgtk2.0-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN /opt/conda/bin/conda update -n base -c defaults conda && \
    /opt/conda/bin/conda install python=3.9 && \
    /opt/conda/bin/conda install anaconda-client && \
    /opt/conda/bin/conda install jupyter -y && \
    /opt/conda/bin/conda install --channel https://conda.anaconda.org/menpo opencv3 -y && \
    /opt/conda/bin/conda install numpy pandas scikit-learn matplotlib pyyaml h5py keras -y && \
    /opt/conda/bin/conda upgrade dask && \
    pip install tensorflow imutils && \
    pip install --no-cache-dir -r requirements.txt

RUN ["mkdir", "dataset"]
COPY conf/.jupyter /root/.jupyter
COPY run_jupyter.sh /

# Jupyter and Tensorboard ports
EXPOSE 8888 6006

# Store notebooks in this mounted directory
VOLUME /dataset

CMD ["/run_jupyter.sh"]