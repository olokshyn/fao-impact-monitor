"""Metaclass helpers for auto-registering concrete subclasses.

Examples
--------
Register ABC subclasses by a string class attribute (``attr``)::

    from abc import ABC, abstractmethod
    from fao_impact_monitor.utils.meta_magic import RegistryMeta

    _PLUGIN_REGISTRY: dict[str, type["Plugin"]] = {}


    class PluginMeta(RegistryMeta):
        registry = _PLUGIN_REGISTRY
        attr = "name"


    class Plugin(ABC, metaclass=PluginMeta):
        name: str

        @abstractmethod
        def run(self) -> None: ...


    class IngestPlugin(Plugin):
        name = "ingest"

        def run(self) -> None: ...


    assert _PLUGIN_REGISTRY["ingest"] is IngestPlugin

Register ``ABC`` + Pydantic ``BaseModel`` subclasses with
``RegistryModelMeta`` (needed so Pydantic's metaclass cooperates)::

    from abc import ABC, abstractmethod
    from pydantic import BaseModel
    from fao_impact_monitor.utils.meta_magic import RegistryModelMeta

    _STAGE_REGISTRY: dict[str, type["StageResult"]] = {}


    class StageResultMeta(RegistryModelMeta):
        registry = _STAGE_REGISTRY
        attr = "name"


    class StageResult(ABC, BaseModel, metaclass=StageResultMeta):
        name: str
        status: str


    class IngestData(StageResult):
        name: str = "ingest"
        status: str = "pending"


    assert _STAGE_REGISTRY["ingest"] is IngestData

Register Beanie inherited documents by ``Settings.class_id_value``::

    from beanie import Document
    from fao_impact_monitor.utils.meta_magic import (
        RegistryModelMeta,
        get_class_id_value,
    )

    _VERSION_REGISTRY: dict[str, type["StageVersion"]] = {}


    class StageVersionMeta(RegistryModelMeta):
        registry = _VERSION_REGISTRY
        use_class_id_value = True


    class StageVersion(Document, metaclass=StageVersionMeta):
        @property
        def name(self) -> str:
            return str(get_class_id_value(type(self)))

        class Settings:
            name = "StageVersion"
            is_root = True
            class_id = "name"


    class IngestVersion(StageVersion):
        class Settings:
            class_id_value = "ingest"


    assert _VERSION_REGISTRY["ingest"] is IngestVersion
"""

from abc import ABCMeta
from typing import Any, ClassVar

from pydantic._internal._model_construction import ModelMetaclass


def get_class_id_value(cls: type[Any]) -> Any:
    """Return Beanie ``Settings.class_id_value`` for an inherited document class."""
    class_id_value = getattr(
        getattr(cls, "Settings", None),
        "class_id_value",
        None,
    )
    if class_id_value is None:
        raise NotImplementedError(f"{cls.__name__} must define Settings.class_id_value")
    return class_id_value


class RegistryMeta(ABCMeta):
    """Register concrete ABC subclasses in ``registry``.

    Configure the metaclass subclass with either:

    * ``attr`` — register from a string class attribute in the class body
    * ``use_class_id_value = True`` — register from ``Settings.class_id_value``

    See the module docstring for full examples.
    """

    registry: ClassVar[dict[str, type[Any]]]
    attr: ClassVar[str | None] = None
    use_class_id_value: ClassVar[bool] = False

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> type:
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        if cls.__dict__.get("__abstractmethods__"):
            return cls

        try:
            registry = mcs.registry
        except AttributeError as exc:
            raise TypeError(
                f"{mcs.__name__} must define class attribute 'registry'"
            ) from exc

        if mcs.use_class_id_value:
            class_id_value = getattr(
                getattr(cls, "Settings", None),
                "class_id_value",
                None,
            )
            if class_id_value is None:
                return cls
            if not isinstance(class_id_value, str):
                raise TypeError(
                    f"Settings.class_id_value must be a string, "
                    f"got {type(class_id_value)} for {name}"
                )
            registry[class_id_value] = cls
            return cls

        attr = mcs.attr
        if attr is None:
            raise TypeError(
                f"{mcs.__name__} must define 'attr' or set use_class_id_value=True"
            )

        if attr not in namespace:
            return cls

        key = namespace[attr]
        if isinstance(key, (property, classmethod, staticmethod)):
            # Base classes may expose ``attr`` as a descriptor (e.g. a property
            # that reads Settings.class_id_value on concrete subclasses).
            return cls
        if not isinstance(key, str):
            raise TypeError(
                f"{attr.capitalize()} must be a string, got {type(key)} for {name}"
            )

        registry[key] = cls
        return cls


class RegistryModelMeta(RegistryMeta, ModelMetaclass):
    """Register concrete ``ABC`` + ``BaseModel`` (or Beanie) subclasses.

    Use this instead of ``RegistryMeta`` when the base also inherits
    ``pydantic.BaseModel`` / Beanie ``Document``. ``RegistryMeta`` is first so
    its ``__new__`` runs, then cooperates with Pydantic via ``super()``.

    See the module docstring for examples (``attr`` and ``use_class_id_value``).
    """
