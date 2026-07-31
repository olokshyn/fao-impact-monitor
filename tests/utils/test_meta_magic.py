from abc import ABC, abstractmethod
from typing import ClassVar

import pytest
from pydantic import BaseModel

from fao_impact_monitor.utils.meta_magic import (
    RegistryMeta,
    RegistryModelMeta,
    get_class_id_value,
)

# --- ABC-only registry -------------------------------------------------------

_PLUGIN_REGISTRY: dict[str, type] = {}


class PluginMeta(RegistryMeta):
    registry = _PLUGIN_REGISTRY
    attr = "name"


class Plugin(ABC, metaclass=PluginMeta):
    name: str

    @abstractmethod
    def run(self) -> str: ...


# --- ABC + BaseModel registry ------------------------------------------------

_STAGE_REGISTRY: dict[str, type] = {}


class ModelStageMeta(RegistryModelMeta):
    registry = _STAGE_REGISTRY
    attr = "name"


class ModelStage(ABC, BaseModel, metaclass=ModelStageMeta):
    name: str

    @abstractmethod
    def run(self) -> str: ...


def test_registry_model_meta_inherits_registry_meta() -> None:
    assert issubclass(RegistryModelMeta, RegistryMeta)


def test_get_class_id_value_reads_settings() -> None:
    class Sample:
        class Settings:
            class_id_value = "sample"

    assert get_class_id_value(Sample) == "sample"


def test_get_class_id_value_requires_settings() -> None:
    class NoSettings:
        pass

    with pytest.raises(
        NotImplementedError, match="must define Settings.class_id_value"
    ):
        get_class_id_value(NoSettings)


def test_attr_only_registers_from_class_attribute() -> None:
    attr_registry: dict[str, type] = {}

    class AttrMeta(RegistryMeta):
        registry = attr_registry
        attr = "name"
        use_class_id_value = False

    class AttrBase(ABC, metaclass=AttrMeta):
        @abstractmethod
        def run(self) -> str: ...

    class AttrConcrete(AttrBase):
        name = "from-attr"

        def run(self) -> str:
            return self.name

    assert attr_registry == {"from-attr": AttrConcrete}
    assert "from-settings" not in attr_registry


def test_attr_only_ignores_settings_class_id_value() -> None:
    attr_registry: dict[str, type] = {}

    class AttrMeta(RegistryMeta):
        registry = attr_registry
        attr = "name"

    class AttrBase(ABC, metaclass=AttrMeta):
        @abstractmethod
        def run(self) -> str: ...

    class SettingsOnly(AttrBase):
        class Settings:
            class_id_value = "from-settings"

        def run(self) -> str:
            return "settings-only"

    assert SettingsOnly not in attr_registry.values()
    assert "from-settings" not in attr_registry


def test_class_id_value_only_registers_from_settings() -> None:
    class_id_registry: dict[str, type] = {}

    class ClassIdMeta(RegistryModelMeta):
        registry = class_id_registry
        use_class_id_value = True

    class ClassIdBase(ABC, BaseModel, metaclass=ClassIdMeta):
        @abstractmethod
        def run(self) -> str: ...

    assert class_id_registry == {}

    class ClassIdConcrete(ClassIdBase):
        class Settings:
            class_id_value = "from-settings"

        def run(self) -> str:
            return str(get_class_id_value(type(self)))

    assert class_id_registry == {"from-settings": ClassIdConcrete}
    assert ClassIdConcrete().run() == "from-settings"


def test_class_id_value_only_ignores_name_class_attribute() -> None:
    class_id_registry: dict[str, type] = {}

    class ClassIdMeta(RegistryModelMeta):
        registry = class_id_registry
        use_class_id_value = True

    class ClassIdBase(ABC, BaseModel, metaclass=ClassIdMeta):
        @abstractmethod
        def run(self) -> str: ...

    class AttrOnly(ClassIdBase):
        name: ClassVar[str] = "from-attr"

        def run(self) -> str:
            return self.name

    assert AttrOnly not in class_id_registry.values()
    assert "from-attr" not in class_id_registry


def test_class_id_value_only_requires_string() -> None:
    class_id_registry: dict[str, type] = {}

    class ClassIdMeta(RegistryModelMeta):
        registry = class_id_registry
        use_class_id_value = True

    class ClassIdBase(ABC, BaseModel, metaclass=ClassIdMeta):
        @abstractmethod
        def run(self) -> str: ...

    with pytest.raises(TypeError, match="Settings.class_id_value must be a string"):

        class BadClassId(ClassIdBase):
            class Settings:
                class_id_value = 123

            def run(self) -> str:
                return "bad"


def test_registry_meta_registers_concrete_abc_subclass() -> None:
    class AlphaPlugin(Plugin):
        name = "alpha"

        def run(self) -> str:
            return "alpha"

    assert _PLUGIN_REGISTRY["alpha"] is AlphaPlugin
    assert AlphaPlugin().run() == "alpha"


def test_registry_meta_skips_abstract_abc_subclass() -> None:
    class AbstractPlugin(Plugin):
        name = "abstract-plugin"

    assert "abstract-plugin" not in _PLUGIN_REGISTRY


def test_registry_meta_skips_annotation_only_key() -> None:
    class AnnotationOnlyPlugin(Plugin):
        name: str

        def run(self) -> str:
            return "noop"

    assert AnnotationOnlyPlugin not in _PLUGIN_REGISTRY.values()


def test_registry_meta_requires_string_key() -> None:
    with pytest.raises(TypeError, match="Name must be a string"):

        class BadPlugin(Plugin):
            name = 123  # type: ignore[assignment]

            def run(self) -> str:
                return "bad"


def test_registry_meta_skips_property_named_like_key() -> None:
    widget_registry: dict[str, type] = {}

    class WidgetMeta(RegistryMeta):
        registry = widget_registry
        attr = "name"

    class Widget(ABC, metaclass=WidgetMeta):
        @property
        def name(self) -> str:
            return "base"

        @abstractmethod
        def run(self) -> str: ...

    assert widget_registry == {}

    class ConcreteWidget(Widget):
        name = "concrete"

        def run(self) -> str:
            return self.name

    assert widget_registry["concrete"] is ConcreteWidget
    assert ConcreteWidget().name == "concrete"


def test_registry_model_meta_registers_concrete_model_subclass() -> None:
    class IngestStage(ModelStage):
        name: str = "ingest"

        def run(self) -> str:
            return "ingest"

    assert _STAGE_REGISTRY["ingest"] is IngestStage
    stage = IngestStage()
    assert isinstance(stage, BaseModel)
    assert stage.name == "ingest"
    assert stage.run() == "ingest"


def test_registry_model_meta_supports_construct_and_model_dump() -> None:
    class ExportStage(ModelStage):
        name: str = "export"
        path: str
        retries: int = 3

        def run(self) -> str:
            return self.path

    stage = ExportStage(path="/tmp/out")
    assert stage.run() == "/tmp/out"
    assert stage.model_dump() == {
        "name": "export",
        "path": "/tmp/out",
        "retries": 3,
    }


def test_registry_model_meta_skips_abstract_model_subclass() -> None:
    class AbstractModelStage(ModelStage):
        name: str = "abstract-stage"

    assert "abstract-stage" not in _STAGE_REGISTRY


def test_registry_model_meta_skips_annotation_only_key() -> None:
    class AnnotationOnlyStage(ModelStage):
        name: str

        def run(self) -> str:
            return "noop"

    assert AnnotationOnlyStage not in _STAGE_REGISTRY.values()


def test_registry_model_meta_requires_string_key() -> None:
    with pytest.raises(TypeError, match="Name must be a string"):

        class BadStage(ModelStage):
            name: int = 123  # type: ignore[assignment]

            def run(self) -> str:
                return "bad"
