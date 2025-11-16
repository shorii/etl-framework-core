import pytest
from pyspark.sql import SparkSession
from typing import Callable
from pyspark.sql import DataFrame
from pandas.testing import assert_frame_equal


@pytest.fixture
def session() -> SparkSession:
    return (
        SparkSession.builder.appName("etl-framework-tests")  # type: ignore
        .master("local[*]")
        .getOrCreate()
    )


@pytest.fixture
def assert_dataframes_equal() -> Callable[[DataFrame, DataFrame], None]:
    def _assert_dataframes_equal(df1: DataFrame, df2: DataFrame):
        pdf1 = df1.toPandas().sort_values(by=df1.columns).reset_index(drop=True)
        pdf2 = df2.toPandas().sort_values(by=df2.columns).reset_index(drop=True)
        assert_frame_equal(pdf1, pdf2)

    return _assert_dataframes_equal
