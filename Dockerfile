FROM tensorflow/tensorflow:1.14.0-gpu-py3

RUN apt-get install -yy libsm-dev libxrender1 libxext6
RUN python3 -m pip install pipenv

COPY Pipfile /src/
COPY Pipfile.lock /src/

WORKDIR /src
RUN pipenv install --dev
