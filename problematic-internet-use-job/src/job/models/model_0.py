"""Problematic Internet Use baseline model definition."""

import logging

import duckdb
from sklearn.pipeline import Pipeline
from models.base_model import BaseModel
from sklearn.dummy import DummyClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class Model0(BaseModel):
    @staticmethod
    def get_model(con: duckdb.DuckDBPyConnection) -> tuple[Pipeline, dict]:
        logging.info("building xgboost classifier")
        x_train = con.execute(
            query='SELECT * EXCLUDE("data_type", "sii") FROM measurements_clean WHERE data_type = ? LIMIT 1',
            parameters=["train"],
        ).fetchdf()
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    [
                        col
                        for col in x_train.columns
                        if x_train[col].dtype not in ["float64", "int64"]
                    ],
                ),
                (
                    "num",
                    StandardScaler(),
                    [
                        col
                        for col in x_train.columns
                        if x_train[col].dtype in ["float64", "int64"]
                    ],
                ),
            ]
        )

        params = {
            "model__strategy": ["prior"],
        }

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", DummyClassifier()),
            ]
        ), params
