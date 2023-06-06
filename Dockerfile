FROM nullland1027/bio:0.0.2-arm64

ADD . /src

ENTRYPOINT ["python", "/src/app.py"]y