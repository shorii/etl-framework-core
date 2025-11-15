from typing import Callable, Dict, List, Optional

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


ExtractorFactory = Callable[[SparkSession], AbstractExtractor]
TransformerFactory = Callable[
    [SparkSession, List[Resource], StructType], AbstractTransformer
]
LoaderFactory = Callable[[SparkSession], AbstractLoader]


class ETLBuilder:
    _registered_extractor_factories: Dict[Resource, ExtractorFactory]
    _registered_transformer_factory: Optional[TransformerFactory]
    _registered_loader_factories: Dict[Resource, LoaderFactory]
    _built: bool

    def __init__(self):
        self._registered_extractor_factories = {}
        self._registered_transformer_factory = None
        self._registered_loader_factories = {}
        self._built = False

    def with_extractor(
        self,
        extractor_factory: ExtractorFactory,
        resource: Resource,
    ) -> "ETLBuilder":
        if resource.storage_type is None:
            raise ValueError("Resource must define storage_type")
        if resource in self._registered_extractor_factories:
            raise ValueError(f"Extractor already registered for resource: {resource}")
        self._registered_extractor_factories[resource] = extractor_factory
        return self

    def with_loader(
        self,
        loader_factory: LoaderFactory,
        resource: Resource,
    ) -> "ETLBuilder":
        if resource.storage_type is None:
            raise ValueError("Resource must define storage_type")
        for output_resource in self._registered_loader_factories:
            if resource.schema != output_resource.schema:
                raise ValueError("All loader resources must have the same schema")
        self._registered_loader_factories[resource] = loader_factory
        return self

    def with_transformer(self, transformer_factory: TransformerFactory) -> "ETLBuilder":
        self._registered_transformer_factory = transformer_factory
        return self

    def _instantiate_extractors(
        self,
        session: SparkSession,
    ) -> Dict[Resource, AbstractExtractor]:
        extractors: Dict[Resource, AbstractExtractor] = {}
        for resource, extractor_factory in self._registered_extractor_factories.items():
            extractor_instance = extractor_factory(session)
            if extractor_instance.STORAGE_TYPE is None:
                raise ValueError("Extractor must define STORAGE_TYPE")
            if extractor_instance.STORAGE_TYPE != resource.storage_type:
                raise ValueError(
                    "Extractor and Resource must have the same storage_type"
                )
            extractors[resource] = extractor_instance
        return extractors

    def _get_input_resources(self) -> List[Resource]:
        input_resources = [
            resource for resource in self._registered_extractor_factories.keys()
        ]
        if not input_resources:
            raise ValueError("At least one extractor resource must be specified")
        return input_resources

    def _get_output_schema(self) -> StructType:
        output_schema = None
        for resource in self._registered_loader_factories.keys():
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
        if self._registered_transformer_factory is None:
            raise ValueError("Transformer class must be registered")
        return self._registered_transformer_factory(
            session,
            input_resources,
            output_schema,
        )

    def _instantiate_loaders(
        self, session: SparkSession
    ) -> Dict[Resource, AbstractLoader]:
        loaders: Dict[Resource, AbstractLoader] = {}
        for resource, loader_factory in self._registered_loader_factories.items():
            loader_instance = loader_factory(session)
            if loader_instance.STORAGE_TYPE is None:
                raise ValueError("Loader must define STORAGE_TYPE")
            if loader_instance.STORAGE_TYPE != resource.storage_type:
                raise ValueError("Loader and Resource must have the same storage_type")
            loaders[resource] = loader_instance
        return loaders

    def build(self, session: SparkSession) -> ETL:
        if self._built:
            raise ValueError("ETL has already been built")

        extractors: Dict[Resource, AbstractExtractor] = self._instantiate_extractors(
            session
        )
        transformer: AbstractTransformer = self._instantiate_transformer(session)
        loaders: Dict[Resource, AbstractLoader] = self._instantiate_loaders(session)

        self._built = True

        return ETL(
            extractors=extractors,
            transformer=transformer,
            loaders=loaders,
        )
