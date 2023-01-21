FROM python:3.9-alpine

ADD . /app
WORKDIR /app
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN conda install -r requirements.txt

ENV FLASK_APP=app.py REDIS_HOST=redis FLASK_RUN_HOST=0.0.0.0

EXPOSE 8888
CMD ["flask", "run"]