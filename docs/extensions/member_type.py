"""Extension adding a jinja2 filter that determines a class member’s type."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from jinja2.defaults import DEFAULT_FILTERS
from jinja2.utils import import_string

if TYPE_CHECKING:
    from sphinx.application import Sphinx


def member_type(obj_path: str) -> Literal["method", "property", "attribute"]:
    """Determine object member type.

    E.g.: `.. auto{{ fullname | member_type }}::`
    """
    # https://jinja.palletsprojects.com/en/stable/api/#custom-filters
    cls_path, member_name = obj_path.rsplit(".", 1)
    cls = import_string(cls_path)
    member = getattr(cls, member_name, None)
    match member:
        case property():
            return "property"
        case _ if callable(member):
            return "method"
        case _:
            return "attribute"


def setup(app: Sphinx):
    """App setup hook."""
    DEFAULT_FILTERS["member_type"] = member_type
