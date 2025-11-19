from abc import ABC, abstractmethod, ABCMeta
from typing import ClassVar, Optional

from pyspark.sql import DataFrame, SparkSession

from .resource import Resource
from .types import StorageType


class LoaderMeta(ABCMeta):
    def __new__(cls, class_name: str, bases: tuple, attrs: dict):
        new_class = super().__new__(cls, class_name, bases, attrs)

        if bool(getattr(new_class, "__abstractmethods__", False)):
            return new_class

        if "STORAGE_TYPE" not in attrs or attrs["STORAGE_TYPE"] is None:
            raise TypeError("STORAGE_TYPE must be defined in subclass")

        return new_class


class AbstractLoader(ABC, metaclass=LoaderMeta):
    STORAGE_TYPE: ClassVar[Optional[StorageType]] = None
    session: SparkSession

    def _setup(self, session: SparkSession) -> None:
        self.session = session

    @abstractmethod
    def load(self, load_as: Resource, content: DataFrame):
        raise NotImplementedError()
