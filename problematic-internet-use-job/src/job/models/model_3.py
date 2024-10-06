"""Problematic Internet Use random forest regressor model definition."""

import logging

import duckdb
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

from models.base_model import BaseModel


class Model3(BaseModel):
    @staticmethod
    def get_model(con: duckdb.DuckDBPyConnection) -> tuple[Pipeline, dict]:
        logging.info(
            "building a voting classifier [xgb classier, random forest classifier]"
        )

        # Fetch a single row to infer column types for preprocessing
        x_train = con.execute(
            query='SELECT * EXCLUDE("data_type", "sii") FROM dataset WHERE data_type = ? LIMIT 1',
            parameters=["train"],
        ).fetchdf()

        # Define a preprocessor for data
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

        # Define the individual regressors
        xgb_reg = XGBClassifier()
        rf_reg = RandomForestClassifier()

        # Create a VotingRegressor
        voting_regressor = VotingClassifier(
            estimators=[("xgb", xgb_reg), ("rf", rf_reg)]
        )

        param_grid = {
            "model__xgb__max_depth": [3, 5, 7],
            "model__xgb__n_estimators": [50, 100, 150],
            "model__rf__max_depth": [None, 10, 20],
            "model__rf__n_estimators": [50, 100, 150],
        }

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", voting_regressor),
            ]
        ), param_grid
