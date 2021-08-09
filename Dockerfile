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


