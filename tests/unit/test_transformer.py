import pytest

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from etl_framework.transformer import TransformContext, AbstractTransformer
from etl_framework.resource import Resource
from typing import Callable


class MyTransformer(AbstractTransformer):
    def process(self, context: TransformContext) -> DataFrame:
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
        df1 = context.get_by_resource(resource1)
        df2 = context.get_by_resource(resource2)
        combined_df = df1.unionByName(df2)
        return combined_df


class TestAbstractTransformer:
    class TestInputValidation:
        @pytest.fixture
        def transformer(self, session: SparkSession) -> MyTransformer:
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
            output_schema = StructType(
                [
                    StructField("id", IntegerType(), False),
                    StructField("name", StringType(), False),
                ]
            )
            return MyTransformer(
                session=session,
                input_resources=[resource1, resource2],
                output_schema=output_schema,
            )

        def test_should_raise_value_error_for_unexpected_resource(
            self,
            transformer: MyTransformer,
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
            resource_dfs = {
                resource1: session.createDataFrame(
                    [(1, "Alice"), (2, "Bob")],
                    schema=resource1.schema,
                ),
                unknown_resource: session.createDataFrame(
                    [(3, "Charlie"), (4, "David")],
                    schema=unknown_resource.schema,
                ),
            }
            with pytest.raises(ValueError):
                transformer._transform(resource_dfs=resource_dfs)

    class TestOutputValidation:
        @pytest.fixture
        def transformer(self, session: SparkSession) -> MyTransformer:
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
            output_schema = StructType(
                [
                    StructField("id", IntegerType(), False),
                    StructField("gender", StringType(), False),
                ]
            )
            return MyTransformer(
                session=session,
                input_resources=[resource1, resource2],
                output_schema=output_schema,
            )

        def test_should_raise_value_error_for_unexpected_resource(
            self,
            transformer: MyTransformer,
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
            with pytest.raises(ValueError):
                transformer._transform(resource_dfs=resource_dfs)

    class TestMatchInputsAndOutput:
        @pytest.fixture
        def transformer(self, session: SparkSession) -> MyTransformer:
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
            output_schema = StructType(
                [
                    StructField("id", IntegerType(), False),
                    StructField("name", StringType(), False),
                ]
            )
            return MyTransformer(
                session=session,
                input_resources=[resource1, resource2],
                output_schema=output_schema,
            )

        def test_should_process_and_return_expected_dataframe(
            self,
            session: SparkSession,
            assert_dataframes_equal: Callable[[DataFrame, DataFrame], None],
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
            output_schema = StructType(
                [
                    StructField("id", IntegerType(), False),
                    StructField("name", StringType(), False),
                ]
            )
            transformer = MyTransformer(
                session=session,
                input_resources=[resource1, resource2],
                output_schema=output_schema,
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
            result_df = transformer._transform(resource_dfs=resource_dfs)
            expected_df = session.createDataFrame(
                [
                    (1, "Alice"),
                    (2, "Bob"),
                    (3, "Charlie"),
                    (4, "David"),
                ],
                schema=output_schema,
            )
            assert_dataframes_equal(result_df, expected_df)
