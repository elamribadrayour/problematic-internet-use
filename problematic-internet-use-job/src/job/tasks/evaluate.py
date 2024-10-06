"""Problematic Internet Use model submission"""

import os
import logging

import duckdb

import helpers.logger
from models.base_model import BaseModel


def get_db(cache_path: str, exec_date: str) -> duckdb.DuckDBPyConnection:
    local_path = os.path.join(cache_path, f"exec_date={exec_date}/type=data")
    os.makedirs(local_path, exist_ok=True)
    return duckdb.connect(database=os.path.join(os.path.join(local_path, "db.duckdb")))


def run(cache_path: str, exec_date: str) -> None:
    helpers.logger.init()
    logging.info(f"evaluating model from {exec_date}")

    con = get_db(cache_path=cache_path, exec_date=exec_date)
    model = BaseModel.load(cache_path=cache_path, exec_date=exec_date)
    logging.info(f"model arch: {model.named_steps['model']}")
    BaseModel.evaluate(model=model, con=con)
