"""Problematic Internet Use api lifespan."""

import os
from contextlib import asynccontextmanager

import joblib
import xgboost
from fastapi import FastAPI
from sklearn.pipeline import Pipeline

from helpers.env import cache_path


model: xgboost.Booster | Pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = get_model()
    yield
    del model


def get_model() -> xgboost.Booster | Pipeline:
    model_path = os.path.join(cache_path, "model/model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    return joblib.load(filename=model_path)
