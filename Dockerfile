FROM python:3.9-buster as build

WORKDIR /var

ADD . /var

RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENV NAME BachelorThesis
ENV PYTHONPATH /var

FROM python:3.9-buster AS run

COPY --from=build /opt/venv /opt/venv

COPY . .

ENV PATH="/opt/venv/bin:$PATH"

RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \
                                        libsndfile1

RUN mkdir "dataset"

CMD ["python", "SvmScript.py"]


