from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from anndata._core.views import _ViewMixin as _AdViewMixin

if TYPE_CHECKING:
    pass


class _ViewMixin(_AdViewMixin):
    """AnnData View Mixin but using ._mudata_ref"""

    # TODO: This makes `deepcopy(obj)` return `obj._view_args.parent._mudata_ref`, fix it
    def __deepcopy__(self, memo):
        parent, attrname, keys = self._view_args
        return deepcopy(getattr(parent._mudata_ref, attrname))


class DictView(_ViewMixin, dict):
    """AnnData DictView adapted for MuData"""

    pass
