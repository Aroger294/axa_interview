FROM python:3.12-slim
LABEL authors="Alistair Rogers"

WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY src/ /app/src
COPY tests/ /app/tests

RUN apt-get update -y && apt install build-essential -y

RUN pip install -r /app/requirements.txt
CMD ["jupyter", "lab", "--allow-root", "--no-browser", "--ip=0.0.0.0", "--NotebookApp.token=''", "--NotebookApp.password=''"]

