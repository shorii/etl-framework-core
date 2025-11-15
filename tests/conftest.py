import pytest
from pyspark.sql import SparkSession


@pytest.fixture
def session() -> SparkSession:
    return (
        SparkSession.builder.appName("etl-framework-tests")  # type: ignore
        .master("local[*]")
        .getOrCreate()
    )
