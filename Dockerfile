FROM python:3.10-bookworm

RUN pip install --upgrade pip

COPY social_action_detector /app/social_action_detector
COPY requirements.txt /app/
COPY setup.py /app/
COPY start_webserver.py /app/
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install -e .


CMD [ "python", "start_webserver.py"]