FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
COPY docker/environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
RUN echo "conda activate bachelorthesis" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure flask is installed:"
RUN python -c "import flask"

# The code to run when container is started:
COPY SvmScript.py entrypoint.sh ./
ENTRYPOINT ["./entrypoint.sh"]