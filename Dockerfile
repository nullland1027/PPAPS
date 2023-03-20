FROM uhub.service.ucloud.cn/openbayesruntimes/python:3.8-py38-cpu.84

ADD . /src
WORKDIR /src

RUN pip install -r requirements.txt
CMD python main.py