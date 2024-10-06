"""Problematic Internet Use base model definition."""

import os
import logging

import pandas
import numpy
import duckdb
import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


class BaseModel:
    @staticmethod
    def get_model(con: duckdb.DuckDBPyConnection) -> tuple[Pipeline, dict]:
        raise NotImplementedError

    @staticmethod
    def train(con: duckdb.DuckDBPyConnection, model: Pipeline, params: dict) -> None:
        x_train = con.execute(
            query="""SELECT * EXCLUDE("sii", "data_type") FROM measurements_clean WHERE data_type = ?""",
            parameters=["train"],
        ).fetchdf()

        y_train = (
            con.execute(
                query='SELECT "sii" FROM measurements_clean WHERE data_type = ?',
                parameters=["train"],
            )
            .fetchdf()
            .squeeze()
        )

        grid_search = GridSearchCV(
            estimator=model, param_grid=params, cv=5, scoring="accuracy", n_jobs=-1
        )
        grid_search.fit(x_train, y_train)
        logging.info(f"Best parameters found: {grid_search.best_params_}")
        logging.info(f"Best cross-validated score: {grid_search.best_score_}")
        model = grid_search.best_estimator_
        model.fit(x_train, y_train)
        return model

    @staticmethod
    def predict(model: Pipeline, x: numpy.ndarray | pandas.DataFrame) -> numpy.ndarray:
        return model.predict(x)

    @staticmethod
    def save(model: Pipeline, cache_path: str, exec_date: str):
        local_path = os.path.join(
            cache_path, f"exec_date={exec_date}/type=model/model.joblib"
        )
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        logging.info(f"saving model to {local_path}")
        with open(local_path, "wb") as f:
            joblib.dump(model, f)

    @staticmethod
    def load(cache_path: str, exec_date: str) -> Pipeline:
        local_path = os.path.join(
            cache_path, f"exec_date={exec_date}/type=model/model.joblib"
        )
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"model doesn't exist in {local_path}")
        logging.info(f"loading model from {local_path}")
        with open(local_path, "rb") as f:
            return joblib.load(f)

    @staticmethod
    def evaluate(model: Pipeline, con: duckdb.DuckDBPyConnection) -> None:
        x_eval = con.execute(
            query="""SELECT * EXCLUDE("sii", "data_type") FROM measurements_clean WHERE data_type = ?""",
            parameters=["eval"],
        ).fetchdf()
        y_pred = BaseModel.predict(model=model, x=x_eval)
        y_eval = (
            con.execute(
                query='SELECT "sii" FROM measurements_clean WHERE data_type = ?',
                parameters=["eval"],
            )
            .fetchdf()
            .squeeze()
        )
        y_pred = BaseModel.predict(model=model, x=x_eval)
        rmse = accuracy_score(y_true=y_eval, y_pred=y_pred)
        logging.info(f"Accuracy: {rmse}")
        return

    @staticmethod
    def test(model: Pipeline, con: duckdb.DuckDBPyConnection) -> numpy.ndarray:
        x_eval = con.execute(
            query="""SELECT * EXCLUDE("id", "sii", "data_type") FROM measurements_clean WHERE data_type = ?""",
            parameters=["test"],
        ).fetchdf()
        return BaseModel.predict(model=model, x=x_eval)
