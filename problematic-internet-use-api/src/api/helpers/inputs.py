"""Problematic Internet Use api input models."""

import pandas
import xgboost
from pydantic import BaseModel


class Input(BaseModel):
    feature_1: float
    feature_2: float

    def to_pandas(self) -> pandas.DataFrame:
        data = [
            [
                self.feature_1,
                self.feature_2
            ]
        ]

        columns = [
            "feature_1",
            "feature_2",
        ]

        output = pandas.DataFrame(data=data, columns=columns)
        return output

    def to_dmatrix(self) -> xgboost.DMatrix:
        return xgboost.DMatrix(self.to_pandas())

    class Config:
        json_schema_extra = {
            "example": {
                "feature_1": 123549.0,
                "feature_2": 17.99,
            }
        }
