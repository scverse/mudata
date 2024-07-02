from copy import deepcopy
from typing import TYPE_CHECKING

from anndata._core.views import ElementRef, _SetItemMixin

if TYPE_CHECKING:
    from .mudata import MuData


class _ViewMixin(_SetItemMixin):
    """
    AnnData View Mixin but using ._mudata_ref
    """

    def __init__(
        self,
        *args,
        view_args: tuple["MuData", str, tuple[str, ...]] = None,
        **kwargs,
    ):
        if view_args is not None:
            view_args = ElementRef(*view_args)
        self._view_args = view_args
        super().__init__(*args, **kwargs)

    # TODO: This makes `deepcopy(obj)` return `obj._view_args.parent._mudata_ref`, fix it
    def __deepcopy__(self, memo):
        parent, attrname, keys = self._view_args
        return deepcopy(getattr(parent._mudata_ref, attrname))


class DictView(_ViewMixin, dict):
    """
    AnnData DictView adopted for MuData
    """

    pass
