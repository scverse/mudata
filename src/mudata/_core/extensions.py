from __future__ import annotations

import inspect
import warnings
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, get_type_hints, overload, runtime_checkable

from .mudata import MuData

if TYPE_CHECKING:
    from collections.abc import Callable


@runtime_checkable
class ExtensionNamespace(Protocol):
    """Protocol for extension namespaces.

    Enforces that the namespace initializer accepts a class with the proper `__init__` method.
    Protocol's can't enforce that the `__init__` accepts the correct types. See
    `_check_namespace_signature` for that. This is mainly useful for static type
    checking with mypy and IDEs.
    """

    def __init__(self, mdata: MuData) -> None:
        """Used to enforce the correct signature for extension namespaces."""


# Based off of the extension framework in Polars
# https://github.com/pola-rs/polars/blob/main/py-polars/polars/api.py

__all__ = ["register_mudata_namespace", "ExtensionNamespace"]


# Reserved namespaces include accessors built into MuData (currently there are none)
# and all current attributes of MuData
_reserved_namespaces: set[str] = set(dir(MuData))

NameSpT = TypeVar("NameSpT", bound=ExtensionNamespace)
T = TypeVar("T")


class AccessorNameSpace(ExtensionNamespace, Generic[NameSpT]):
    """Establish property-like namespace object for user-defined functionality."""

    def __init__(self, name: str, namespace: type[NameSpT]) -> None:
        self._accessor = name
        self._ns = namespace

    @overload
    def __get__(self, instance: None, cls: type[T]) -> type[NameSpT]: ...

    @overload
    def __get__(self, instance: T, cls: type[T]) -> NameSpT: ...

    def __get__(self, instance: T | None, cls: type[T]) -> NameSpT | type[NameSpT]:
        if instance is None:
            return self._ns

        ns_instance = self._ns(instance)  # type: ignore[call-arg]
        setattr(instance, self._accessor, ns_instance)
        return ns_instance


def _check_namespace_signature(ns_class: type) -> None:
    """Validate the signature of a namespace class for MuData extensions.

    This function ensures that any class intended to be used as an extension namespace
    has a properly formatted `__init__` method such that:

    1. Accepts at least two parameters (self and mdata)
    2. Has 'mdata' as the name of the second parameter
    3. Has the second parameter properly type-annotated as 'MuData' or any equivalent import alias

    The function performs runtime validation of these requirements before a namespace
    can be registered through the `register_mudata_namespace` decorator.

    Parameters
    ----------
    ns_class
        The namespace class to validate.

    Raises
    ------
        TypeError
            If the `__init__` method has fewer than 2 parameters (missing the MuData parameter).
        AttributeError
            If the second parameter of `__init__` lacks a type annotation.
        TypeError
            If the second parameter of `__init__` is not named 'mdata'.
        TypeError
            If the second parameter of `__init__` is not annotated as the 'MuData' class.
        TypeError
            If both the name and type annotation of the second parameter are incorrect.

    """
    sig = inspect.signature(ns_class.__init__)
    params = sig.parameters

    # Ensure there are at least two parameters (self and mdata)
    if len(params) < 2:
        raise TypeError("Namespace initializer must accept an MuData instance as the second parameter.")

    # Get the second parameter (expected to be 'mdata')
    param = iter(params.values())
    next(param)
    param = next(param)
    if param.annotation is inspect.Parameter.empty:
        raise AttributeError(
            "Namespace initializer's second parameter must be annotated as the 'MuData' class, got empty annotation."
        )

    name_ok = param.name == "mdata"

    # Resolve the annotation using get_type_hints to handle forward references and aliases.
    try:
        type_hints = get_type_hints(ns_class.__init__)
        resolved_type = type_hints.get(param.name, param.annotation)
    except NameError as e:
        raise NameError(f"Namespace initializer's second parameter must be named 'mdata', got '{param.name}'.") from e

    type_ok = resolved_type is MuData

    match (name_ok, type_ok):
        case (True, True):
            return  # Signature is correct.
        case (False, True):
            raise TypeError(f"Namespace initializer's second parameter must be named 'mdata', got {param.name!r}.")
        case (True, False):
            type_repr = getattr(resolved_type, "__name__", str(resolved_type))
            raise TypeError(
                f"Namespace initializer's second parameter must be annotated as the 'MuData' class, got {type_repr!r}."
            )
        case _:
            type_repr = getattr(resolved_type, "__name__", str(resolved_type))
            raise TypeError(
                f"Namespace initializer's second parameter must be named 'mdata', got {param.name!r}. "
                f"And must be annotated as 'MuData', got {type_repr!r}."
            )


def _create_namespace(name: str, cls: type[MuData]) -> Callable[[type[NameSpT]], type[NameSpT]]:
    """Register custom namespace against the underlying MuData class."""

    def namespace(ns_class: type[NameSpT]) -> type[NameSpT]:
        _check_namespace_signature(ns_class)  # Perform the runtime signature check
        if name in _reserved_namespaces:
            raise AttributeError(f"cannot override reserved attribute {name!r}")
        elif hasattr(cls, name):
            warnings.warn(
                f"Overriding existing custom namespace {name!r} (on {cls.__name__!r})", UserWarning, stacklevel=2
            )
        setattr(cls, name, AccessorNameSpace(name, ns_class))
        return ns_class

    return namespace


def register_mudata_namespace(name: str) -> Callable[[type[NameSpT]], type[NameSpT]]:
    """Decorator for registering custom functionality with an :class:`~mudata.MuData` object.

    This decorator allows you to extend MuData objects with custom methods and properties
    organized under a namespace. The namespace becomes accessible as an attribute on MuData
    instances, providing a clean way to you to add domain-specific functionality without modifying
    the MuData class itself, or extending the class with additional methods as you see fit in your workflow.

    This is equivalent to :func:`anndata.register_anndata_namespace`.

    Parameters
    ----------
    name
        Name under which the accessor should be registered. This will be the attribute name
        used to access your namespace's functionality on MuData objects (e.g., `mdata.{name}`).
        Cannot conflict with existing MuData attributes like `obs`, `var`, `mod`, etc. The list of reserved
        attributes includes everything outputted by `dir(MuData)`.

    Returns
    -------
    A decorator that registers the decorated class as a custom namespace.

    Notes
    -----
    Implementation requirements:

    1. The decorated class must have an `__init__` method that accepts exactly one parameter
       (besides `self`) named `mdata` and annotated with type :class:`~mudata.MuData`.
    2. The namespace will be initialized with the MuData object on first access and then
       cached on the instance.
    3. If the namespace name conflicts with an existing namespace, a warning is issued.
    4. If the namespace name conflicts with a built-in MuData attribute, an AttributeError is raised.
    """
    return _create_namespace(name, MuData)
