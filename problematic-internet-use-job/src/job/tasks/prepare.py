"""prepare data for Problematic Internet Use."""

import os
import pandas
import logging
from zipfile import ZipFile

import duckdb
from sklearn.model_selection import train_test_split

import helpers.logger


def get_db(cache_path: str, exec_date: str) -> duckdb.DuckDBPyConnection:
    local_path = os.path.join(cache_path, f"exec_date={exec_date}/type=data")
    os.makedirs(local_path, exist_ok=True)
    return duckdb.connect(database=os.path.join(os.path.join(local_path, "db.duckdb")))


def get_unzipped_data(cache_path: str) -> None:
    zip_path = os.path.join(
        cache_path, "child-mind-institute-problematic-internet-use.zip"
    )
    if os.path.exists(zip_path) is False:
        raise FileNotFoundError(
            "you need to download the data file from kaggle and put in problematic-internet-use-cache directory"
        )

    logging.info(f"file_path={zip_path}")
    with ZipFile(file=zip_path) as ref:
        ref.extractall(path=cache_path)


def get_raw_data(cache_path: str, con: duckdb.DuckDBPyConnection, force: bool) -> None:
    local_paths_csv = [
        os.path.join(cache_path, f"{file_name}.csv") for file_name in ["train", "test"]
    ]
    local_dirs_parquet = [
        os.path.join(cache_path, f"series_{file_name}.parquet")
        for file_name in ["train", "test"]
    ]
    if any(
        os.path.exists(local_path) is False
        for local_path in local_dirs_parquet + local_paths_csv
    ):
        get_unzipped_data(cache_path=cache_path)

    logging.info(f"reading csv files: {local_paths_csv}")
    condition = "OR REPLACE TABLE" if force is True else "TABLE IF NOT EXISTS"
    query = f"""
        CREATE {condition} measurements AS
        SELECT * FROM read_csv({local_paths_csv}, union_by_name = true, filename = true);
    """
    con.execute(query=query)

    local_paths_parquet = [
        os.path.join(dir_name, "**/*.parquet") for dir_name in local_dirs_parquet
    ]
    logging.info(f"reading parquet files from directories: {local_paths_parquet}")
    query = f"""
        CREATE {condition} accelerometer AS
        SELECT * FROM read_parquet({local_paths_parquet}, hive_partitioning=true);
    """
    con.execute(query=query)


def get_clean_data(con: duckdb.DuckDBPyConnection) -> None:
    logging.info("cleaning measurements data & building measurements clean table")

    dataframe = con.execute(
        query="""
        SELECT
            id,
            ARRAY_AGG(step ORDER BY step ASC) AS step,
            ARRAY_AGG(X ORDER BY step ASC) AS X,
            ARRAY_AGG(Y ORDER BY step ASC) AS Y,
            ARRAY_AGG(Z ORDER BY step ASC) AS Z,
            ARRAY_AGG(enmo ORDER BY step ASC) AS enmo,
            ARRAY_AGG(anglez ORDER BY step ASC) AS anglez,
            ARRAY_AGG(light ORDER BY step ASC) AS light,
            ARRAY_AGG(battery_voltage ORDER BY step ASC) AS battery_voltage,
            ARRAY_AGG(time_of_day ORDER BY step ASC) AS time_of_day,
            ARRAY_AGG(weekday ORDER BY step ASC) AS weekday,
            ARRAY_AGG(quarter ORDER BY step ASC) AS quarter,
            ARRAY_AGG(relative_date_PCIAT ORDER BY step ASC) AS relative_date_PCIAT,
            ARRAY_AGG("non-wear_flag" ORDER BY step ASC) AS "non-wear_flag"
        FROM accelerometer AS a
        WHERE id = '00115b9f'
        GROUP BY id
        LIMIT 1;
        """,
    ).fetch_df()
    logging.info(dataframe)
    dataframe.to_json("/code/data/sample.jsonl", lines=True, orient="records")
    exit(0)

    logging.info("creating data type column")
    dataframe["data_type"] = dataframe["filename"].apply(
        lambda x: x.split("/")[-1].split(".")[0]
    )
    dataframe.drop(columns=["filename"], inplace=True)

    logging.info("processing categorical columns")
    categorical_columns = [
        "FGC-Season",
        "BIA-Season",
        "SDS-Season",
        "CGAS-Season",
        "PAQ_A-Season",
        "PAQ_C-Season",
        "Basic_Demos-Sex",
        "FGC-FGC_CU_Zone",
        "FGC-FGC_PU_Zone",
        "Physical-Season",
        "FGC-FGC_TL_Zone",
        "FGC-FGC_GSD_Zone",
        "FGC-FGC_SRL_Zone",
        "FGC-FGC_SRR_Zone",
        "FGC-FGC_GSND_Zone",
        "BIA-BIA_Frame_num",
        "PreInt_EduHx-Season",
        "Basic_Demos-Enroll_Season",
        "Fitness_Endurance-Season",
        "BIA-BIA_Activity_Level_num",
        "PreInt_EduHx-computerinternet_hoursday",
    ]
    for column in categorical_columns:
        if dataframe[column].dtype.name in ["int64", "float64"]:
            dataframe[column] = dataframe[column].fillna(value=-1).astype(int)
        else:
            dataframe[column] = dataframe[column].fillna(value="unknown")
        dataframe[column] = dataframe[column].astype("category")

    logging.info("processing numerical columns")
    numerical_columns = [
        "sii",
        "FGC-FGC_PU",
        "FGC-FGC_CU",
        "FGC-FGC_TL",
        "FGC-FGC_SRL",
        "FGC-FGC_SRR",
        "BIA-BIA_BMC",
        "BIA-BIA_BMI",
        "BIA-BIA_BMR",
        "BIA-BIA_DEE",
        "BIA-BIA_ECW",
        "BIA-BIA_FFM",
        "BIA-BIA_FMI",
        "BIA-BIA_Fat",
        "BIA-BIA_ICW",
        "BIA-BIA_LDM",
        "BIA-BIA_LST",
        "BIA-BIA_SMM",
        "BIA-BIA_TBW",
        "FGC-FGC_GSD",
        "FGC-FGC_GSND",
        "BIA-BIA_FFMI",
        "Physical-BMI",
        "SDS-SDS_Total_T",
        "Basic_Demos-Age",
        "CGAS-CGAS_Score",
        "Physical-Height",
        "Physical-Weight",
        "PAQ_A-PAQ_A_Total",
        "PAQ_C-PAQ_C_Total",
        "SDS-SDS_Total_Raw",
        "Physical-HeartRate",
        "Physical-Systolic_BP",
        "Physical-Diastolic_BP",
        "Fitness_Endurance-Time_Sec",
        "Fitness_Endurance-Max_Stage",
        "Fitness_Endurance-Time_Mins",
        "Physical-Waist_Circumference",
    ]
    for column in numerical_columns:
        dataframe[column] = dataframe[column].astype(float)
        dataframe[column] = pandas.to_numeric(dataframe[column], errors="coerce")

    test_set = dataframe.query("data_type == 'test'")
    train_set = dataframe.query("data_type == 'train'").dropna(subset=["sii"])
    train_set, eval_set = train_test_split(train_set, test_size=0.2, random_state=42)
    eval_set["data_type"] = "eval"
    dataframe = pandas.concat([train_set, eval_set, test_set])
    con.execute("CREATE OR REPLACE TABLE dataset AS SELECT * FROM dataframe")

    return


def run(cache_path: str, exec_date: str, force: bool) -> None:
    helpers.logger.init()
    logging.info("running data preparation for Problematic Internet Use")
    con = get_db(cache_path=cache_path, exec_date=exec_date)
    get_raw_data(con=con, cache_path=cache_path, force=force)
    get_clean_data(con=con)
    logging.info("data successfully prepared")
