from abc import ABC, abstractmethod, ABCMeta
from typing import ClassVar, Optional

from pyspark.sql import DataFrame, SparkSession

from .resource import Resource
from .types import StorageType


class ExtractorMeta(ABCMeta):
    def __new__(cls, class_name: str, bases: tuple, attrs: dict):
        new_class = super().__new__(cls, class_name, bases, attrs)

        if bool(getattr(new_class, "__abstractmethods__", False)):
            return new_class

        if "STORAGE_TYPE" not in attrs or attrs["STORAGE_TYPE"] is None:
            raise TypeError("STORAGE_TYPE must be defined in subclass")

        return new_class


class AbstractExtractor(ABC, metaclass=ExtractorMeta):
    STORAGE_TYPE: ClassVar[Optional[StorageType]] = None
    session: SparkSession

    def _setup(self, session: SparkSession) -> None:
        self.session = session

    @abstractmethod
    def extract(self, resource: Resource) -> DataFrame:
        raise NotImplementedError()
