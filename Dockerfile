FROM gcr.io/kaggle-gpu-images/python:v120

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN pip install -U pip && \
    pip install fastprogress japanize-matplotlib && \
    pip install -q h5py
