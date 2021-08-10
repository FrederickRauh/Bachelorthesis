FROM ubuntu:latest

WORKDIR /app

COPY docker/requirements.txt .
COPY docker/environment.yml .

RUN apt-get update \
    && apt-get install -y python3-pip python3-dev \
    && cd /usr/local/bin \
    && ln -s /usr/bin/python3 python \
    && pip3 install --upgrade pip \
    && pip install -r requirements.txt \

COPY SvmScript.py .
COPY GmmScript.py .

CMD ["python3", "SvmScript.py"]
CMD ["python3", "GmmScript.py"]