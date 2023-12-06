FROM python:3.10-bookworm

RUN pip install --upgrade pip
ARG HUGGINGFACE_TOKEN
ENV HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN

COPY social_action_detector /app/social_action_detector
COPY requirements.txt /app/
COPY setup.py /app/
COPY start_webserver.py /app/
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install -e .
RUN huggingface-cli login --token $HUGGINGFACE_TOKEN && huggingface-cli download 'JeanMachado/social_good_detector'

CMD [ "python", "start_webserver.py"]