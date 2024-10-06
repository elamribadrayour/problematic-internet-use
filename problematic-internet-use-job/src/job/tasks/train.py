"""Problematic Internet Use model training"""

import os

import duckdb

import helpers.logger

from models.model_0 import Model0  # noqa F401
from models.model_1 import Model1  # noqa F401
from models.model_2 import Model2  # noqa F401
from models.model_3 import Model3  # noqa F401


def get_db(cache_path: str, exec_date: str) -> duckdb.DuckDBPyConnection:
    local_path = os.path.join(cache_path, f"exec_date={exec_date}/type=data")
    os.makedirs(local_path, exist_ok=True)
    return duckdb.connect(database=os.path.join(os.path.join(local_path, "db.duckdb")))


def run(cache_path: str, exec_date: str) -> None:
    helpers.logger.init()
    con = get_db(cache_path=cache_path, exec_date=exec_date)
    model, params = Model3.get_model(con=con)
    model = Model3.train(
        con=con,
        model=model,
        params=params,
    )
    Model3.save(model=model, cache_path=cache_path, exec_date=exec_date)
