"""Problematic Internet Use main file."""

from typing import Annotated

from typer import Typer, Argument


import tasks.train
import tasks.submit
import tasks.prepare
import tasks.evaluate


app = Typer()


@app.command()
def prepare(
    force: Annotated[bool, Argument(envvar="FORCE")],
    exec_date: Annotated[str, Argument(envvar="EXEC_DATE")],
    cache_path: Annotated[str, Argument(envvar="CACHE_PATH")],
) -> None:
    tasks.prepare.run(
        force=force,
        exec_date=exec_date,
        cache_path=cache_path,
    )


@app.command()
def train(
    exec_date: Annotated[str, Argument(envvar="EXEC_DATE")],
    cache_path: Annotated[str, Argument(envvar="CACHE_PATH")],
) -> None:
    tasks.train.run(cache_path=cache_path, exec_date=exec_date)


@app.command()
def evaluate(
    exec_date: Annotated[str, Argument(envvar="EXEC_DATE")],
    cache_path: Annotated[str, Argument(envvar="CACHE_PATH")],
) -> None:
    tasks.evaluate.run(cache_path=cache_path, exec_date=exec_date)


@app.command()
def submit(
    exec_date: Annotated[str, Argument(envvar="EXEC_DATE")],
    cache_path: Annotated[str, Argument(envvar="CACHE_PATH")],
) -> None:
    tasks.submit.run(cache_path=cache_path, exec_date=exec_date)


if __name__ == "__main__":
    app()
