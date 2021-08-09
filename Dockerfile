FROM continuumio/anaconda3
MAINTAINER "Frederick Rauh"

WORKDIR /var

COPY environment.yml .

RUN conda env create -f environment.yml

RUN conda activate bachelorthesis
RUN echo "Here we go:"

# The code to run when container is started:
ENTRYPOINT ["python", "SvmScript.py"]