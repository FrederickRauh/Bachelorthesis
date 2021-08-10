FROM continuumio/miniconda
WORKDIR /usr/src/app
COPY ./ ./
RUN conda env create -f docker/environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "bachelorthesis", "/bin/bash", "-c"]

RUN apt-get --yes install libsndfile1

EXPOSE 5003
# The code to run when container is started:
ENTRYPOINT ["conda", "run", "-n", "bachelorthesis", "python3", "SvmScript.py"]