FROM tensorflow/tensorflow:2.1.0-py3

RUN pip install tensorflow-datasets==2.0.0

RUN mkdir -p /app
ADD mnist-dist.py /app/

