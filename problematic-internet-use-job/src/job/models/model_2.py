"""Problematic Internet Use random forest regressor model definition."""

import logging

import duckdb
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from models.base_model import BaseModel


class Model2(BaseModel):
    @staticmethod
    def get_model(con: duckdb.DuckDBPyConnection) -> tuple[Pipeline, dict]:
        logging.info("building xgboost classifier")
        x_train = con.execute(
            query='SELECT * EXCLUDE("data_type", "sii") FROM dataset WHERE data_type = ? LIMIT 1',
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
            "model__max_depth": [None, 10, 20],
            "model__n_estimators": [50, 100, 200],
        }

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", XGBClassifier()),
            ]
        ), params
