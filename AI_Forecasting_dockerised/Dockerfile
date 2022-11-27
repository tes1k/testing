FROM python:3.10.8

COPY ./docker_app/requirements.txt .
RUN pip3 install -r requirements.txt

RUN mkdir -p /tmp/docker_app
COPY . .
WORKDIR /docker_app/

RUN echo 'finish'