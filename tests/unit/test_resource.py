from pyspark.sql.types import DateType, IntegerType, StringType, StructField, StructType

from etl_framework_core.resource import Resource


class TestResource:
    class TestEquality:
        def test_resources_are_the_same(self):
            resource = Resource(
                location="/path/to/dummy.csv",
                schema=StructType(
                    [
                        StructField("string_field", StringType(), False),
                        StructField("integer_field", IntegerType(), False),
                        StructField("date_field", DateType(), False),
                    ]
                ),
                storage_type="local_storage",
            )
            other = Resource(
                location="/path/to/dummy.csv",
                schema=StructType(
                    [
                        StructField("string_field", StringType(), False),
                        StructField("integer_field", IntegerType(), False),
                        StructField("date_field", DateType(), False),
                    ]
                ),
                storage_type="local_storage",
            )
            assert resource == other

        def test_resources_have_different_location(self):
            resource = Resource(
                location="/path/to/dummy.parquet",
                schema=StructType(
                    [
                        StructField("string_field", StringType(), False),
                        StructField("integer_field", IntegerType(), False),
                        StructField("date_field", DateType(), False),
                    ]
                ),
                storage_type="local_storage",
            )
            other = Resource(
                location="/path/to/dummy.csv",
                schema=StructType(
                    [
                        StructField("string_field", StringType(), False),
                        StructField("integer_field", IntegerType(), False),
                        StructField("date_field", DateType(), False),
                    ]
                ),
                storage_type="local_storage",
            )
            assert resource != other

        def test_resources_have_different_schema(self):
            resource = Resource(
                location="/path/to/dummy.csv",
                schema=StructType(
                    [
                        StructField("different_string_field", StringType(), False),
                        StructField("integer_field", IntegerType(), False),
                        StructField("date_field", DateType(), False),
                    ]
                ),
                storage_type="local_storage",
            )
            other = Resource(
                location="/path/to/dummy.csv",
                schema=StructType(
                    [
                        StructField("string_field", StringType(), False),
                        StructField("integer_field", IntegerType(), False),
                        StructField("date_field", DateType(), False),
                    ]
                ),
                storage_type="local_storage",
            )
            assert resource != other

        def test_resources_have_different_storage_type(self):
            resource = Resource(
                location="/path/to/dummy.csv",
                schema=StructType(
                    [
                        StructField("string_field", StringType(), False),
                        StructField("integer_field", IntegerType(), False),
                        StructField("date_field", DateType(), False),
                    ]
                ),
                storage_type="local_storage",
            )
            other = Resource(
                location="/path/to/dummy.csv",
                schema=StructType(
                    [
                        StructField("string_field", StringType(), False),
                        StructField("integer_field", IntegerType(), False),
                        StructField("date_field", DateType(), False),
                    ]
                ),
                storage_type="remote_storage",
            )
            assert resource != other
