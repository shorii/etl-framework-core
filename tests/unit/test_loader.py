from etl_framework.loader import AbstractLoader
from etl_framework.resource import Resource
from pyspark.sql import DataFrame
import pytest


class TestAbstractExtractor:
    class TestStorageTypeValidation:
        def test_should_raise_type_error_if_storage_type_not_defined(self):
            class InvalidLoader(AbstractLoader):
                def load(self, load_as: Resource, content: DataFrame):
                    pass

            with pytest.raises(
                TypeError, match="STORAGE_TYPE must be defined in subclass"
            ):
                InvalidLoader()

        def test_should_not_raise_type_error_if_storage_type_defined(self):
            class ValidLoader(AbstractLoader):
                STORAGE_TYPE = "local"

                def load(self, load_as: Resource, content: DataFrame):
                    pass

            # Should not raise any exception
            ValidLoader()
