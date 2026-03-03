from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pytest

import mudata as md
from mudata._core import extensions

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def _cleanup_dummy() -> Generator[None, None, None]:
    """Automatically cleanup dummy namespace after each test."""
    original = getattr(md.MuData, "dummy", None)
    yield
    if original is not None:
        md.MuData.dummy = original
    elif hasattr(md.MuData, "dummy"):
        delattr(md.MuData, "dummy")


@pytest.fixture
def dummy_namespace() -> type:
    """Create a basic dummy namespace class."""

    @md.register_mudata_namespace("dummy")
    class DummyNamespace:
        def __init__(self, mdata: md.MuData) -> None:
            self._mdata = mdata

        def greet(self) -> str:
            return "hello"

    return DummyNamespace


@pytest.fixture
def mdata(rng) -> md.MuData:
    """Create a basic MuData object for testing."""
    return md.MuData({"test": ad.AnnData(X=rng.poisson(1, size=(10, 10)))})


def test_accessor_namespace() -> None:
    """Test the behavior of the AccessorNameSpace descriptor.

    This test verifies that:
    - When accessed at the class level (i.e., without an instance), the descriptor
      returns the namespace type.
    - When accessed via an instance, the descriptor instantiates the namespace,
      passing the instance to its constructor.
    - The instantiated namespace is then cached on the instance such that subsequent
      accesses of the same attribute return the cached namespace instance.
    """

    # Define a dummy namespace class to be used via the descriptor.
    class DummyNamespace:
        def __init__(self, mdata: md.MuData) -> None:
            self._mdata = mdata

        def foo(self) -> str:
            return "foo"

    class Dummy:
        pass

    descriptor = extensions.AccessorNameSpace("dummy", DummyNamespace)

    # When accessed on the class, it should return the namespace type.
    ns_class = descriptor.__get__(None, Dummy)
    assert ns_class is DummyNamespace

    # When accessed via an instance, it should instantiate DummyNamespace.
    dummy_obj = Dummy()
    ns_instance = descriptor.__get__(dummy_obj, Dummy)
    assert isinstance(ns_instance, DummyNamespace)
    assert ns_instance._mdata is dummy_obj

    # __get__ should cache the namespace instance on the object.
    # Subsequent access should return the same cached instance.
    assert dummy_obj.dummy is ns_instance


def test_descriptor_instance_caching(dummy_namespace: type, mdata: md.MuData) -> None:
    """Test that namespace instances are cached on individual MuData objects."""
    # First access creates the instance
    ns_instance = mdata.dummy
    # Subsequent accesses should return the same instance
    assert mdata.dummy is ns_instance


def test_register_namespace_basic(dummy_namespace: type, mdata: md.MuData) -> None:
    """Test basic namespace registration and access."""
    assert mdata.dummy.greet() == "hello"


def test_register_namespace_override(dummy_namespace: type) -> None:
    """Test namespace registration and override behavior."""
    assert hasattr(md.MuData, "dummy")

    # Override should warn and update the namespace
    with pytest.warns(UserWarning, match="Overriding existing custom namespace 'dummy'"):

        @md.register_mudata_namespace("dummy")
        class DummyNamespaceOverride:
            def __init__(self, mdata: md.MuData) -> None:
                self._mdata = mdata

            def greet(self) -> str:
                return "world"

    # Verify the override worked
    mdata = md.MuData({"test": ad.AnnData(X=np.random.poisson(1, size=(10, 10)))})
    assert mdata.dummy.greet() == "world"


@pytest.mark.parametrize(
    "attr",
    [
        "mod",
        "obs",
        "var",
        "uns",
        "obsm",
        "varm",
        "copy",
        "write",
        "obsmap",
        "varmap",
        "obsp",
        "varp",
        "update",
        "update_obs",
        "update_var",
        "push_obs",
        "push_var",
        "pull_obs",
        "pull_var",
    ],
)
def test_register_existing_attributes(attr: str) -> None:
    """
    Test that registering an accessor with a name that is a reserved attribute of MuData raises an attribute error.

    We only test a representative sample of important attributes rather than all of them.
    """
    # Test a representative sample of key AnnData attributes
    with pytest.raises(AttributeError, match=f"cannot override reserved attribute {attr!r}"):

        @md.register_mudata_namespace(attr)
        class DummyNamespace:
            def __init__(self, mdata: md.MuData) -> None:
                self._mdata = mdata


def test_valid_signature() -> None:
    """Test that a namespace with valid signature is accepted."""

    @md.register_mudata_namespace("valid")
    class ValidNamespace:
        def __init__(self, mdata: md.MuData) -> None:
            self.mdata = mdata


def test_missing_param() -> None:
    """Test that a namespace missing the second parameter is rejected."""
    with pytest.raises(
        TypeError, match=r"Namespace initializer must accept an MuData instance as the second parameter\."
    ):

        @md.register_mudata_namespace("missing_param")
        class MissingParamNamespace:
            def __init__(self) -> None:
                pass


def test_wrong_name() -> None:
    """Test that a namespace with wrong parameter name is rejected."""
    with pytest.raises(
        TypeError, match=r"Namespace initializer's second parameter must be named 'mdata', got 'notmdata'\."
    ):

        @md.register_mudata_namespace("wrong_name")
        class WrongNameNamespace:
            def __init__(self, notmdata: md.MuData) -> None:
                self.notmdata = notmdata


def test_wrong_annotation() -> None:
    """Test that a namespace with wrong parameter annotation is rejected."""
    with pytest.raises(
        TypeError,
        match=r"Namespace initializer's second parameter must be annotated as the 'MuData' class, got 'int'\.",
    ):

        @md.register_mudata_namespace("wrong_annotation")
        class WrongAnnotationNamespace:
            def __init__(self, mdata: int) -> None:
                self.mdata = mdata


def test_missing_annotation() -> None:
    """Test that a namespace with missing parameter annotation is rejected."""
    with pytest.raises(AttributeError):

        @md.register_mudata_namespace("missing_annotation")
        class MissingAnnotationNamespace:
            def __init__(self, mdata) -> None:
                self.mdata = mdata


def test_both_wrong() -> None:
    """Test that a namespace with both wrong name and annotation is rejected."""
    with pytest.raises(
        TypeError,
        match=(
            r"Namespace initializer's second parameter must be named 'mdata', got 'info'\. "
            r"And must be annotated as 'MuData', got 'str'\."
        ),
    ):

        @md.register_mudata_namespace("both_wrong")
        class BothWrongNamespace:
            def __init__(self, info: str) -> None:
                self.info = info
