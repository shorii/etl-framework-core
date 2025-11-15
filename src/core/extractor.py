from abc import ABC, abstractmethod
from typing import ClassVar, Optional

from pyspark.sql import DataFrame

from .resource import Resource
from .types import StorageType


class AbstractExtractor(ABC):
    STORAGE_TYPE: ClassVar[Optional[StorageType]] = None

    def __init__(self):
        if not hasattr(self, "STORAGE_TYPE") or self.STORAGE_TYPE is None:
            raise TypeError("STORAGE_TYPE must be defined in subclass")
        super().__init__()

    @abstractmethod
    def extract(self, resource: Resource) -> DataFrame:
        raise NotImplementedError()
