"""Problematic Internet Use model submission"""

import os
import logging
import subprocess


import duckdb

import helpers.logger
from models.base_model import BaseModel


def get_db(cache_path: str, exec_date: str) -> duckdb.DuckDBPyConnection:
    local_path = os.path.join(cache_path, f"exec_date={exec_date}/type=data")
    os.makedirs(local_path, exist_ok=True)
    return duckdb.connect(database=os.path.join(os.path.join(local_path, "db.duckdb")))


def run(cache_path: str, exec_date: str) -> None:
    helpers.logger.init()
    logging.info("submitting result to kaggle")

    con = get_db(cache_path=cache_path, exec_date=exec_date)
    model = BaseModel.load(cache_path=cache_path, exec_date=exec_date)
    BaseModel.test(model=model, con=con)

    file_path = os.path.join(cache_path, "submissions.csv")
    competition = "child-mind-institute-problematic-internet-use"
    command = f"kaggle competitions submit -c {competition} -f {file_path}"
    result = subprocess.run(command.split(), capture_output=True, text=True)

    if result.returncode == 0:
        logging.info(f"subcessfully submitted {file_path}")
    else:
        logging.info(f"submition of {file_path} failed")
