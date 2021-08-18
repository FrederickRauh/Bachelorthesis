FROM continuumio/miniconda
WORKDIR /var/src
COPY . .
RUN conda env create -f docker/environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "bachelorthesis", "/bin/bash", "-c"]

RUN pip install -r docker/requirements.txt

RUN apt-get --yes install libsndfile1
RUN apt-get --yes install libportaudio2

VOLUME /dataset

EXPOSE 5003
# The code to run when container is started:
ENTRYPOINT ["conda", "run", "-n", "bachelorthesis", "python3", "main.py"]
