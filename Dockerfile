FROM python:3.10-bookworm

COPY . /app
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -e .


CMD ["python", "handler.py"]