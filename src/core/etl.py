from typing import Dict, List, Optional, Type

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType

from .extractor import AbstractExtractor
from .loader import AbstractLoader
from .resource import Resource
from .transformer import AbstractTransformer


class ETL:
    _extractors: Dict[Resource, AbstractExtractor]
    _transformer: AbstractTransformer
    _loaders: Dict[Resource, AbstractLoader]

    def __init__(
        self,
        extractors: Dict[Resource, AbstractExtractor],
        transformer: AbstractTransformer,
        loaders: Dict[Resource, AbstractLoader],
    ):
        self._extractors = extractors
        self._transformer = transformer
        self._loaders = loaders

    def run(self):
        extracted_dfs: Dict[Resource, DataFrame] = {}
        for resource, extractor in self._extractors.items():
            if resource in extracted_dfs:
                raise ValueError(f"Conflicting resource: {resource._location}")
            df = extractor.extract(resource)
            extracted_dfs[resource] = df

        content: DataFrame = self._transformer._transform(extracted_dfs)

        for resource, loader in self._loaders.items():
            if resource.schema != content.schema:
                raise ValueError(
                    f"Schema mismatch for loader resource: {resource._location}"
                )
            loader.load(load_as=resource, content=content)


class ETLBuilder:
    _registered_extractor_classes: Dict[Resource, Type[AbstractExtractor]]
    _registered_transformer_class: Optional[Type[AbstractTransformer]]
    _registered_loader_classes: Dict[Resource, Type[AbstractLoader]]
    _built: bool

    def __init__(self):
        self._registered_extractor_classes = {}
        self._registered_transformer_class = None
        self._registered_loader_classes = {}
        self._built = False

    def with_extractor(
        self,
        extractor_class: Type[AbstractExtractor],
        resource: Resource,
    ) -> "ETLBuilder":
        if extractor_class.STORAGE_TYPE is None:
            raise ValueError("Extractor type must define STORAGE_TYPE")
        if resource.storage_type is None:
            raise ValueError("Resource must define storage_type")
        if extractor_class.STORAGE_TYPE != resource.storage_type:
            raise ValueError(
                "Extractor type STORAGE_TYPE does not match resource storage_type"
            )
        if resource in self._registered_extractor_classes:
            raise ValueError(f"Extractor already registered for resource: {resource}")

        self._registered_extractor_classes[resource] = extractor_class

        return self

    def with_loader(
        self,
        loader_class: Type[AbstractLoader],
        resource: Resource,
    ) -> "ETLBuilder":
        if loader_class.STORAGE_TYPE is None:
            raise ValueError("Loader type must define STORAGE_TYPE")
        if resource.storage_type is None:
            raise ValueError("Resource must define storage_type")
        if loader_class.STORAGE_TYPE != resource.storage_type:
            raise ValueError(
                "Loader type STORAGE_TYPE does not match resource storage_type"
            )
        for output_resource in self._registered_loader_classes.keys():
            if resource.schema != output_resource.schema:
                raise ValueError("All loader resources must have the same schema")

        self._registered_loader_classes[resource] = loader_class

        return self

    def with_transformer(self, transformer: Type[AbstractTransformer]) -> "ETLBuilder":
        self._registered_transformer_class = transformer
        return self

    def _instantiate_extractors(self) -> Dict[Resource, AbstractExtractor]:
        extractors: Dict[Resource, AbstractExtractor] = {}
        for resource, extractor_class in self._registered_extractor_classes.items():
            extractor_instance = extractor_class()
            extractors[resource] = extractor_instance
        return extractors

    def _get_input_resources(self) -> List[Resource]:
        input_resources = [
            resource for resource in self._registered_extractor_classes.keys()
        ]
        if not input_resources:
            raise ValueError("At least one extractor resource must be specified")
        return input_resources

    def _get_output_schema(self) -> StructType:
        output_schema = None
        for resource in self._registered_loader_classes.keys():
            if output_schema is None:
                output_schema = resource.schema
                continue
            if resource.schema == output_schema:
                continue
            raise ValueError("All loader resources must have the same schema")

        if output_schema is None:
            raise ValueError("At least one loader resource must be specified")
        return output_schema

    def _instantiate_transformer(self, session: SparkSession) -> AbstractTransformer:
        input_resources = self._get_input_resources()
        output_schema = self._get_output_schema()
        if self._registered_transformer_class is None:
            raise ValueError("Transformer class must be registered")
        return self._registered_transformer_class(
            session=session,
            input_resources=input_resources,
            output_schema=output_schema,
        )

    def _instantiate_loaders(self) -> Dict[Resource, AbstractLoader]:
        loaders: Dict[Resource, AbstractLoader] = {}
        for resource, loader_class in self._registered_loader_classes.items():
            loader_instance = loader_class()
            loaders[resource] = loader_instance
        return loaders

    def build(self, session: SparkSession) -> ETL:
        if self._built:
            raise ValueError("ETL has already been built")

        extractors: Dict[Resource, AbstractExtractor] = self._instantiate_extractors()
        transformer: AbstractTransformer = self._instantiate_transformer(session)
        loaders: Dict[Resource, AbstractLoader] = self._instantiate_loaders()

        self._built = True

        return ETL(
            extractors=extractors,
            transformer=transformer,
            loaders=loaders,
        )
