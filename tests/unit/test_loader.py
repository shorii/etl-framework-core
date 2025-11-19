from etl_framework_core.loader import AbstractLoader
from etl_framework_core.resource import Resource
from pyspark.sql import DataFrame
import pytest


class TestAbstractExtractor:
    class TestStorageTypeValidation:
        def test_should_raise_type_error_if_storage_type_not_defined(self):
            with pytest.raises(
                TypeError, match="STORAGE_TYPE must be defined in subclass"
            ):

                class InvalidLoader(AbstractLoader):
                    def load(self, load_as: Resource, content: DataFrame):
                        pass

        def test_should_not_raise_type_error_if_storage_type_defined(self):
            class ValidLoader(AbstractLoader):
                STORAGE_TYPE = "local"

                def load(self, load_as: Resource, content: DataFrame):
                    pass

            # Should not raise any exception
            ValidLoader()
