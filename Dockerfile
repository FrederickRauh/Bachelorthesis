FROM python:3.9
EXPOSE 8501
WORKDIR /app
COPY . .
RUN pip install -r ./docker/requirements.txt
CMD streamlit run SvmScript.py