FROM python:3.9-slim-buster

WORKDIR /app

ADD . /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENV NAME BachelorThesis
ENV PYTHONPATH /app

RUN mkdir "dataset"

CMD ["python", ".SvmScript.py"]


