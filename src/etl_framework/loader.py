from abc import ABC, abstractmethod
from typing import ClassVar, Optional

from pyspark.sql import DataFrame, SparkSession

from .resource import Resource
from .types import StorageType


class AbstractLoader(ABC):
    STORAGE_TYPE: ClassVar[Optional[StorageType]] = None
    session: SparkSession

    def __init__(self):
        if not hasattr(self, "STORAGE_TYPE") or self.STORAGE_TYPE is None:
            raise TypeError("STORAGE_TYPE must be defined in subclass")
        super().__init__()

    def _setup(self, session: SparkSession) -> None:
        self.session = session

    @abstractmethod
    def load(self, load_as: Resource, content: DataFrame):
        raise NotImplementedError()
