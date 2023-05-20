FROM arm64v8/python:3.8-bullseye

ADD . /src
WORKDIR /src

RUN pip install -r requirements.txt
CMD python app.py