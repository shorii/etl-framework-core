from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from etl_framework.etl import ETL, ETLBuilder
from etl_framework.resource import Resource
from etl_framework.extractor import AbstractExtractor
from etl_framework.loader import AbstractLoader
from etl_framework.transformer import AbstractTransformer, TransformContext
from typing import Type, Callable
import pytest
import re


class LocalExtractor(AbstractExtractor):
    STORAGE_TYPE = "local"

    def extract(self, resource: Resource) -> DataFrame:
        return self.session.createDataFrame([], schema=resource.schema)


class LocalLoader(AbstractLoader):
    STORAGE_TYPE = "local"

    def __init__(self):
        self.content_save_as = None
        self.loaded_content_df = None
        super().__init__()

    def load(self, load_as: Resource, content: DataFrame):
        self.content_save_as = load_as
        self.loaded_content_df = content


class LocalResource(Resource):
    def __init__(self, path: str, schema: StructType):
        super().__init__(
            location=path,
            schema=schema,
            storage_type="local",
        )


class RemoteExtractor(AbstractExtractor):
    STORAGE_TYPE = "remote"

    def extract(self, resource: Resource) -> DataFrame:
        return self.session.createDataFrame([], schema=resource.schema)


class RemoteLoader(AbstractLoader):
    STORAGE_TYPE = "remote"

    def load(self, load_as: Resource, content: DataFrame):
        self.content_save_as = load_as
        self.loaded_content_df = content
        super().__init__()


class RemoteResource(Resource):
    def __init__(self, path: str, schema: StructType):
        super().__init__(
            location=path,
            schema=schema,
            storage_type="remote",
        )


class NoneStorageTypeExtractor(AbstractExtractor):
    def extract(self, resource: Resource) -> DataFrame:
        return self.session.createDataFrame([], schema=resource.schema)


class NoneStorageTypeLoader(AbstractLoader):
    def __init__(self):
        # explicitly not calling super().__init__() for testing purposes
        pass

    def load(self, load_as: Resource, content: DataFrame):
        pass


class NoneStorageTypeResource(Resource):
    def __init__(self, path: str, schema: StructType):
        # storage_type is explicitly set to None
        super().__init__(
            location=path,
            schema=schema,
            storage_type=None,  # type: ignore
        )


class TestETLBuilder:
    @pytest.fixture
    def none_storage_type_resource(
        self,
    ) -> NoneStorageTypeResource:
        return NoneStorageTypeResource(
            path="/path/to/resource",
            schema=StructType(
                [
                    StructField("id", IntegerType(), False),
                    StructField("name", StringType(), False),
                ]
            ),
        )

    @pytest.fixture
    def input_local_resource(self) -> LocalResource:
        return LocalResource(
            path="/path/to/input_local_resource",
            schema=StructType(
                [
                    StructField("id", IntegerType(), False),
                    StructField("name", StringType(), False),
                ]
            ),
        )

    @pytest.fixture
    def input_remote_resource(self) -> RemoteResource:
        return RemoteResource(
            path="/path/to/input_remote_resource",
            schema=StructType(
                [
                    StructField("id", IntegerType(), False),
                    StructField("name", StringType(), False),
                ]
            ),
        )

    @pytest.fixture
    def output_local_resource(self) -> LocalResource:
        return LocalResource(
            path="/path/to/output_local_resource",
            schema=StructType(
                [
                    StructField("id", IntegerType(), False),
                    StructField("name", StringType(), False),
                ]
            ),
        )

    @pytest.fixture
    def output_remote_resource(self) -> RemoteResource:
        return RemoteResource(
            path="/path/to/output_remote_resource",
            schema=StructType(
                [
                    StructField("id", IntegerType(), False),
                    StructField("name", StringType(), False),
                ]
            ),
        )

    @pytest.fixture
    def transformer_class(
        self, output_local_resource: Resource
    ) -> Type[AbstractTransformer]:
        class Transformer(AbstractTransformer):
            def transform(
                self,
                context: TransformContext,
            ) -> DataFrame:
                return context.create_empty(output_local_resource.schema)

        return Transformer

    class TestExtractorRegistration:
        def test_should_raise_value_error_if_resource_storage_type_not_defined(self):
            etl_builder = ETLBuilder()
            with pytest.raises(ValueError, match="Resource must define storage_type"):
                etl_builder.with_extractor(
                    extractor_factory=lambda: NoneStorageTypeExtractor(),
                    resource=NoneStorageTypeResource(
                        path="/path/to/resource",
                        schema=StructType(
                            [
                                StructField("id", IntegerType(), False),
                                StructField("name", StringType(), False),
                            ]
                        ),
                    ),
                )

        def test_should_raise_value_error_if_extractor_already_registered(
            self, input_local_resource: LocalResource
        ):
            etl_builder = ETLBuilder()
            with pytest.raises(
                ValueError,
                match=re.escape(
                    f"Extractor already registered for resource: {input_local_resource}"
                ),
            ):
                etl_builder.with_extractor(
                    extractor_factory=lambda: LocalExtractor(),
                    resource=input_local_resource,
                ).with_extractor(
                    extractor_factory=lambda: LocalExtractor(),
                    resource=input_local_resource,
                )

        def test_should_register_extractor_successfully(
            self,
            input_local_resource: LocalResource,
            input_remote_resource: RemoteResource,
        ):
            etl_builder = ETLBuilder()
            etl_builder.with_extractor(
                extractor_factory=lambda: LocalExtractor(),
                resource=input_local_resource,
            ).with_extractor(
                extractor_factory=lambda: RemoteExtractor(),
                resource=input_remote_resource,
            )
            assert len(etl_builder._registered_extractor_factories) == 2
            assert input_local_resource in etl_builder._registered_extractor_factories
            assert input_remote_resource in etl_builder._registered_extractor_factories

    class TestLoaderRegistration:
        def test_should_raise_value_error_if_resource_storage_type_not_defined(self):
            etl_builder = ETLBuilder()
            with pytest.raises(ValueError, match="Resource must define storage_type"):
                etl_builder.with_loader(
                    loader_factory=lambda: NoneStorageTypeLoader(),
                    resource=NoneStorageTypeResource(
                        path="/path/to/resource",
                        schema=StructType(
                            [
                                StructField("id", IntegerType(), False),
                                StructField("name", StringType(), False),
                            ]
                        ),
                    ),
                )

        def test_should_raise_value_error_if_loader_already_registered(
            self, output_local_resource: LocalResource
        ):
            etl_builder = ETLBuilder()
            with pytest.raises(
                ValueError,
                match=re.escape(
                    f"Loader already registered for resource: {output_local_resource}"
                ),
            ):
                etl_builder.with_loader(
                    loader_factory=lambda: LocalLoader(),
                    resource=output_local_resource,
                ).with_loader(
                    loader_factory=lambda: LocalLoader(),
                    resource=output_local_resource,
                )

        def test_should_raise_value_error_if_loader_resources_have_different_schemas(
            self,
            output_local_resource: LocalResource,
        ):
            etl_builder = ETLBuilder()
            with pytest.raises(
                ValueError,
                match="All loader resources must have the same schema",
            ):
                etl_builder.with_loader(
                    loader_factory=lambda: LocalLoader(),
                    resource=output_local_resource,
                ).with_loader(
                    loader_factory=lambda: LocalLoader(),
                    resource=LocalResource(
                        path="/path/to/different_schema_resource",
                        schema=StructType(
                            [
                                StructField("id", IntegerType(), False),
                                StructField("age", IntegerType(), False),
                            ]
                        ),
                    ),
                )

        def test_should_register_loader_successfully(
            self,
            output_local_resource: LocalResource,
        ):
            etl_builder = ETLBuilder()
            another_local_output_resource = LocalResource(
                path="/path/to/another_output_resource",
                schema=output_local_resource.schema,
            )
            remote_output_resource = RemoteResource(
                path="/path/to/remote_output_resource",
                schema=output_local_resource.schema,
            )
            etl_builder.with_loader(
                loader_factory=lambda: LocalLoader(),
                resource=output_local_resource,
            ).with_loader(
                loader_factory=lambda: LocalLoader(),
                resource=another_local_output_resource,
            ).with_loader(
                loader_factory=lambda: RemoteLoader(),
                resource=remote_output_resource,
            )
            assert len(etl_builder._registered_loader_factories) == 3
            assert output_local_resource in etl_builder._registered_loader_factories
            assert (
                another_local_output_resource
                in etl_builder._registered_loader_factories
            )
            assert remote_output_resource in etl_builder._registered_loader_factories

    class TestTransformerRegistration:
        def test_should_register_transformer_successfully(
            self,
            transformer_class: Type[AbstractTransformer],
        ):
            etl_builder = ETLBuilder()
            transformer_factory = lambda: transformer_class()
            etl_builder.with_transformer(transformer_factory=transformer_factory)
            assert etl_builder._registered_transformer_factory is transformer_factory

    class TestBuild:
        def test_should_build_etl_successfully(
            self,
            session: SparkSession,
            input_local_resource: LocalResource,
            output_local_resource: LocalResource,
            transformer_class: Type[AbstractTransformer],
        ):
            etl_builder = ETLBuilder()
            transformer_factory = lambda: transformer_class()
            etl_instance = (
                etl_builder.with_extractor(
                    extractor_factory=lambda: LocalExtractor(),
                    resource=input_local_resource,
                )
                .with_loader(
                    loader_factory=lambda: LocalLoader(),
                    resource=output_local_resource,
                )
                .with_transformer(transformer_factory=transformer_factory)
                .build(session)
            )
            assert isinstance(etl_instance, ETL)

        def test_should_raise_value_error_if_no_extractor_registered(
            self,
            session: SparkSession,
            output_local_resource: LocalResource,
            transformer_class: Type[AbstractTransformer],
        ):
            etl_builder = ETLBuilder()
            transformer_factory = lambda: transformer_class()
            with pytest.raises(
                ValueError,
                match="At least one extractor resource must be specified",
            ):
                (
                    etl_builder.with_loader(
                        loader_factory=lambda: LocalLoader(),
                        resource=output_local_resource,
                    )
                    .with_transformer(transformer_factory=transformer_factory)
                    .build(session)
                )

        def test_should_raise_value_error_if_no_loader_registered(
            self,
            session: SparkSession,
            input_local_resource: LocalResource,
            transformer_class: Type[AbstractTransformer],
        ):
            etl_builder = ETLBuilder()
            transformer_factory = lambda: transformer_class()
            with pytest.raises(
                ValueError,
                match="At least one loader resource must be specified",
            ):
                (
                    etl_builder.with_extractor(
                        extractor_factory=lambda: LocalExtractor(),
                        resource=input_local_resource,
                    )
                    .with_transformer(transformer_factory=transformer_factory)
                    .build(session)
                )

        def test_should_raise_value_error_if_no_transformer_registered(
            self,
            session: SparkSession,
            input_local_resource: LocalResource,
            output_local_resource: LocalResource,
        ):
            etl_builder = ETLBuilder()
            with pytest.raises(
                ValueError,
                match="Transformer class must be registered",
            ):
                (
                    etl_builder.with_extractor(
                        extractor_factory=lambda: LocalExtractor(),
                        resource=input_local_resource,
                    )
                    .with_loader(
                        loader_factory=lambda: LocalLoader(),
                        resource=output_local_resource,
                    )
                    .build(session)
                )

        def test_should_raise_value_error_if_extractor_storage_type_mismatch(
            self,
            session: SparkSession,
            input_remote_resource: RemoteResource,
            output_local_resource: LocalResource,
            transformer_class: Type[AbstractTransformer],
        ):
            etl_builder = ETLBuilder()
            transformer_factory = lambda: transformer_class()
            with pytest.raises(
                ValueError,
                match="Extractor and Resource must have the same storage_type",
            ):
                (
                    etl_builder.with_extractor(
                        extractor_factory=lambda: LocalExtractor(),
                        resource=input_remote_resource,
                    )
                    .with_loader(
                        loader_factory=lambda: LocalLoader(),
                        resource=output_local_resource,
                    )
                    .with_transformer(transformer_factory=transformer_factory)
                    .build(session)
                )

        def test_should_raise_value_error_if_loader_storage_type_mismatch(
            self,
            session: SparkSession,
            input_local_resource: LocalResource,
            output_remote_resource: RemoteResource,
            transformer_class: Type[AbstractTransformer],
        ):
            etl_builder = ETLBuilder()
            transformer_factory = lambda: transformer_class()
            with pytest.raises(
                ValueError,
                match="Loader and Resource must have the same storage_type",
            ):
                (
                    etl_builder.with_extractor(
                        extractor_factory=lambda: LocalExtractor(),
                        resource=input_local_resource,
                    )
                    .with_loader(
                        loader_factory=lambda: LocalLoader(),
                        resource=output_remote_resource,
                    )
                    .with_transformer(transformer_factory=transformer_factory)
                    .build(session)
                )

        def test_should_raise_value_error_if_build_called_multiple_times(
            self,
            session: SparkSession,
            input_local_resource: LocalResource,
            output_local_resource: LocalResource,
            transformer_class: Type[AbstractTransformer],
        ):
            etl_builder = ETLBuilder()
            transformer_factory = lambda: transformer_class()
            etl_instance = (
                etl_builder.with_extractor(
                    extractor_factory=lambda: LocalExtractor(),
                    resource=input_local_resource,
                )
                .with_loader(
                    loader_factory=lambda: LocalLoader(),
                    resource=output_local_resource,
                )
                .with_transformer(transformer_factory=transformer_factory)
                .build(session)
            )
            assert isinstance(etl_instance, ETL)

            with pytest.raises(ValueError, match="ETL has already been built"):
                etl_builder.build(session)


class TestETL:
    @pytest.fixture
    def transformer_class(self, output_resource: Resource) -> Type[AbstractTransformer]:
        class Transformer(AbstractTransformer):
            def transform(
                self,
                context: TransformContext,
            ) -> DataFrame:
                return context.create_empty(output_resource.schema)

        return Transformer

    @pytest.fixture
    def input_local_resource(self) -> LocalResource:
        return LocalResource(
            path="/path/to/input_resource",
            schema=StructType(
                [
                    StructField("id", IntegerType(), False),
                    StructField("name", StringType(), False),
                ]
            ),
        )

    @pytest.fixture
    def output_local_resource(self) -> LocalResource:
        return LocalResource(
            path="/path/to/output_resource",
            schema=StructType(
                [
                    StructField("id", IntegerType(), False),
                    StructField("name", StringType(), False),
                ]
            ),
        )

    class TestRun:
        def test_should_run_etl_successfully(
            self,
            session: SparkSession,
            input_local_resource: LocalResource,
            output_local_resource: LocalResource,
            transformer_class: Type[AbstractTransformer],
            assert_dataframes_equal: Callable[[DataFrame, DataFrame], None],
        ):
            loader = LocalLoader()
            etl_instance = (
                ETLBuilder()
                .with_extractor(
                    extractor_factory=lambda: LocalExtractor(),
                    resource=input_local_resource,
                )
                .with_loader(
                    loader_factory=lambda: loader,
                    resource=output_local_resource,
                )
                .with_transformer(transformer_factory=lambda: transformer_class())
                .build(session)
            )

            etl_instance.run()

            assert loader.loaded_content_df is not None
            assert loader.content_save_as is not None
            expected_df = session.createDataFrame(
                [], schema=output_local_resource.schema
            )
            assert_dataframes_equal(loader.loaded_content_df, expected_df)
            assert loader.content_save_as == output_local_resource
            assert loader.content_save_as.schema == loader.loaded_content_df.schema

        class TestShouldRaiseValueErrorIfLoaderSchemaMismatch:
            @pytest.fixture
            def transformer_class(self) -> Type[AbstractTransformer]:
                class MismatchedSchemaTransformer(AbstractTransformer):
                    def transform(
                        self,
                        context: TransformContext,
                    ) -> DataFrame:
                        # Returning a DataFrame with a schema that does not match output_resource.schema
                        mismatched_schema = StructType(
                            [
                                StructField("id", IntegerType(), False),
                                StructField("age", IntegerType(), False),
                            ]
                        )
                        return context.create_empty(mismatched_schema)

                return MismatchedSchemaTransformer

            def test_should_raise_value_error_if_loader_schema_mismatch(
                self,
                session: SparkSession,
                input_local_resource: LocalResource,
                output_local_resource: LocalResource,
                transformer_class: Type[AbstractTransformer],
            ):
                etl_instance = (
                    ETLBuilder()
                    .with_extractor(
                        extractor_factory=lambda: LocalExtractor(),
                        resource=input_local_resource,
                    )
                    .with_loader(
                        loader_factory=lambda: LocalLoader(),
                        resource=output_local_resource,
                    )
                    .with_transformer(transformer_factory=lambda: transformer_class())
                    .build(session)
                )

                with pytest.raises(
                    ValueError,
                    match=re.escape(
                        f"Schema mismatch for loader resource: {output_local_resource.location}"
                    ),
                ):
                    etl_instance.run()

        # class TestShouldRaiseValueErrorIfConflictingResource:
        #     def test_should_raise_value_error_if_conflicting_resource(
        #         self,
        #         session: SparkSession,
        #         transformer_class: Type[AbstractTransformer],
        #         output_resource: LocalResource,
        #     ):
        #         input_resource = LocalResource(
        #             path="/path/to/conflicting_resource",
        #             schema=StructType(
        #                 [
        #                     StructField("id", IntegerType(), False),
        #                     StructField("name", StringType(), False),
        #                 ]
        #             ),
        #         )
        #         etl_instance = (
        #             ETLBuilder()
        #             .with_extractor(
        #                 extractor_factory=lambda: LocalExtractor(),
        #                 resource=input_resource,
        #             )
        #             .with_extractor(
        #                 extractor_factory=lambda: LocalExtractor(),
        #                 resource=input_resource,  # same resource to create conflict
        #             )
        #             .with_loader(
        #                 loader_factory=lambda: LocalLoader(),
        #                 resource=output_resource,
        #             )
        #             .with_transformer(transformer_factory=lambda: transformer_class())
        #             .build(session)
        #         )

        #         with pytest.raises(
        #             ValueError,
        #             match=re.escape(f"Conflicting resource: {input_resource.location}"),
        #         ):
        #             etl_instance.run()
