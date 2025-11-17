from abc import ABC, abstractmethod
from typing import Dict, List

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType

from .resource import Resource


class TransformContext:
    _session: SparkSession
    _resource_dfs: Dict[Resource, DataFrame]

    def __init__(self, session: SparkSession, resource_dfs: Dict[Resource, DataFrame]):
        self._session = session
        self._resource_dfs = resource_dfs

    def get_by_resource(self, resource: Resource) -> DataFrame:
        return self._resource_dfs[resource]

    def create_empty(self, schema: StructType) -> DataFrame:
        return self._session.createDataFrame([], schema)


class AbstractTransformer(ABC):
    def __init__(
        self,
        session: SparkSession,
        input_resources: List[Resource],
        output_schema: StructType,
    ):
        self._session = session
        self._input_resources = input_resources
        self._output_schema = output_schema

    def _validate_input_resources(self, resource_dfs: Dict[Resource, DataFrame]):
        for resource in resource_dfs:
            if resource not in self._input_resources:
                raise ValueError(
                    f"Resource mismatch for transformer: {resource._location}"
                )

    def _transform(self, resource_dfs: Dict[Resource, DataFrame]) -> DataFrame:
        self._validate_input_resources(resource_dfs)
        context = TransformContext(
            session=self._session,
            resource_dfs=resource_dfs,
        )
        transformed_df: DataFrame = self.transform(context)
        if transformed_df.schema != self._output_schema:
            raise ValueError("Output schema does not match expected schema.")
        return transformed_df

    @abstractmethod
    def transform(self, context: TransformContext) -> DataFrame:
        raise NotImplementedError()
