from dataclasses import dataclass

from pyspark.sql.types import StructType

from .types import StorageType


@dataclass(frozen=True)
class Resource:
    location: str
    schema: StructType
    storage_type: StorageType

    def __eq__(self, other):
        if not isinstance(other, Resource):
            return False
        return (
            self.location == other.location
            and self.schema == other.schema
            and self.storage_type == other.storage_type
        )

    def __hash__(self):
        return hash((self.location, self.schema.simpleString(), self.storage_type))
