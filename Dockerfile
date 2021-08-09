FROM python:3.9

WORKDIR /app

ADD . /app

ENV NAME BachelorThesis
ENV PYTHONPATH /app

RUN ls

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["python", "scripts/SvmScript.py"]


