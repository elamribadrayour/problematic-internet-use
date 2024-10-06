FROM python:3.11.8-slim

RUN mkdir -p /code/api /code/data
WORKDIR /code/api


RUN apt update && apt install gcc -y && apt clean
RUN pip install --upgrade pip && pip install poetry
RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock /code/api/
RUN poetry install --without ci --no-root

COPY src/api /code/api
