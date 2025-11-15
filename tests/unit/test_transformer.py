import pytest

from datetime import date
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType
from etl_framework.transformer import TransformContext
from etl_framework.resource import Resource
from pandas.testing import assert_frame_equal


def assert_dataframes_equal(df1: DataFrame, df2: DataFrame):
    pdf1 = df1.toPandas().sort_values(by=df1.columns).reset_index(drop=True)
    pdf2 = df2.toPandas().sort_values(by=df2.columns).reset_index(drop=True)
    assert_frame_equal(pdf1, pdf2)


class TestTransformContext:
    class TestGetByResource:
        def test_should_return_dataframe_for_resource(
            self,
            session: SparkSession,
        ):
            resource1 = Resource(
                _location="/path/to/resource1",
                schema=StructType(
                    [
                        StructField("id", IntegerType(), False),
                        StructField("name", StringType(), False),
                    ]
                ),
                storage_type="local",
            )
            resource2 = Resource(
                _location="/path/to/resource2",
                schema=StructType(
                    [
                        StructField("id", IntegerType(), False),
                        StructField("name", StringType(), False),
                    ]
                ),
                storage_type="local",
            )
            resource3 = Resource(
                _location="/path/to/resource1",
                schema=StructType(
                    [
                        StructField("id", IntegerType(), False),
                        StructField("date", DateType(), False),
                    ]
                ),
                storage_type="local",
            )
            resource4 = Resource(
                _location="/path/to/resource1",
                schema=StructType(
                    [
                        StructField("id", IntegerType(), False),
                        StructField("name", StringType(), False),
                    ]
                ),
                storage_type="remote",
            )

            resource_dfs = {
                resource1: session.createDataFrame(
                    [(1, "Alice"), (2, "Bob")],
                    schema=resource1.schema,
                ),
                resource2: session.createDataFrame(
                    [(3, "Charlie"), (4, "David")],
                    schema=resource2.schema,
                ),
                resource3: session.createDataFrame(
                    [(1, date(2023, 1, 1)), (2, date(2023, 1, 2))],
                    schema=resource3.schema,
                ),
                resource4: session.createDataFrame(
                    [(5, "Eve"), (6, "Frank")],
                    schema=resource4.schema,
                ),
            }
            context = TransformContext(session=session, resource_dfs=resource_dfs)

            expected_df_of_resource1 = context.get_by_resource(resource1)
            assert_dataframes_equal(
                expected_df_of_resource1,
                session.createDataFrame(
                    [(1, "Alice"), (2, "Bob")],
                    schema=StructType(
                        [
                            StructField("id", IntegerType(), False),
                            StructField("name", StringType(), False),
                        ],
                    ),
                ),
            )
            expected_df_of_resource2 = context.get_by_resource(resource2)
            assert_dataframes_equal(
                expected_df_of_resource2,
                session.createDataFrame(
                    [(3, "Charlie"), (4, "David")],
                    schema=StructType(
                        [
                            StructField("id", IntegerType(), False),
                            StructField("name", StringType(), False),
                        ],
                    ),
                ),
            )
            expected_df_of_resource3 = context.get_by_resource(resource3)
            assert_dataframes_equal(
                expected_df_of_resource3,
                session.createDataFrame(
                    [(1, date(2023, 1, 1)), (2, date(2023, 1, 2))],
                    schema=StructType(
                        [
                            StructField("id", IntegerType(), False),
                            StructField("date", DateType(), False),
                        ],
                    ),
                ),
            )
            expected_df_of_resource4 = context.get_by_resource(resource4)
            assert_dataframes_equal(
                expected_df_of_resource4,
                session.createDataFrame(
                    [(5, "Eve"), (6, "Frank")],
                    schema=StructType(
                        [
                            StructField("id", IntegerType(), False),
                            StructField("name", StringType(), False),
                        ],
                    ),
                ),
            )

        def test_should_raise_key_error_for_unknown_resource(
            self, session: SparkSession
        ):
            resource1 = Resource(
                _location="/path/to/resource1",
                schema=StructType(
                    [
                        StructField("id", IntegerType(), False),
                        StructField("name", StringType(), False),
                    ]
                ),
                storage_type="local",
            )
            resource2 = Resource(
                _location="/path/to/resource2",
                schema=StructType(
                    [
                        StructField("id", IntegerType(), False),
                        StructField("name", StringType(), False),
                    ]
                ),
                storage_type="local",
            )

            resource_dfs = {
                resource1: session.createDataFrame(
                    [(1, "Alice"), (2, "Bob")],
                    schema=resource1.schema,
                ),
                resource2: session.createDataFrame(
                    [(3, "Charlie"), (4, "David")],
                    schema=resource2.schema,
                ),
            }
            context = TransformContext(session=session, resource_dfs=resource_dfs)
            unknown_resource = Resource(
                _location="/path/to/unknown_resource",
                schema=StructType(
                    [
                        StructField("id", IntegerType(), False),
                        StructField("name", StringType(), False),
                    ]
                ),
                storage_type="local",
            )
            with pytest.raises(KeyError):
                context.get_by_resource(unknown_resource)

    class TestCreateEmpty:
        def test_should_create_empty_dataframe_with_given_schema(
            self,
            session: SparkSession,
        ):
            schema = StructType(
                [
                    StructField("id", IntegerType(), False),
                    StructField("name", StringType(), False),
                    StructField("date", DateType(), False),
                ]
            )
            context = TransformContext(session=session, resource_dfs={})

            empty_df = context.create_empty(schema=schema)

            assert empty_df.count() == 0
            assert empty_df.schema == schema
