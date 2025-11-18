from etl_framework.extractor import AbstractExtractor
from etl_framework.resource import Resource
from pyspark.sql import DataFrame, SparkSession
import pytest


class TestAbstractExtractor:
    class TestStorageTypeValidation:
        def test_should_raise_type_error_if_storage_type_not_defined(
            self, session: SparkSession
        ):
            with pytest.raises(
                TypeError, match="STORAGE_TYPE must be defined in subclass"
            ):

                class InvalidExtractor(AbstractExtractor):
                    def __init__(self, session: SparkSession):
                        self._session = session
                        super().__init__()

                    def extract(self, resource: Resource) -> DataFrame:
                        return self._session.createDataFrame([], schema=resource.schema)

        def test_should_not_raise_type_error_if_storage_type_defined(
            self, session: SparkSession
        ):
            class ValidExtractor(AbstractExtractor):
                STORAGE_TYPE = "local"

                def __init__(self, session: SparkSession):
                    self._session = session
                    super().__init__()

                def extract(self, resource: Resource) -> DataFrame:
                    return self._session.createDataFrame([], schema=resource.schema)

            # Should not raise any exception
            ValidExtractor(session)
