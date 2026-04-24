from typing import Annotated, Literal

from pydantic import Field
from scverse_misc import Settings


class _Settings(Settings, exported_object_name="settings", docstring_style="numpy"):
    display_style: Literal["text", "html"] = "text"
    """:class:`MuData` object representation to use in notebooks."""

    display_html_expand: Annotated[int, Field(ge=0, le=4)] = 0x2
    """3-bit flag influencing whether to expand MuData slots when using HTML representation.

       - 0x4: expand MuData slots
       - 0x2: expand `.mod` slots
       - 0x1: expand slots for each modality
    """

    pull_on_update: bool = False
    """Whether to pull columns from modalities into the global `.obs`/`.var` when running :meth:`MuData.update`."""


settings = _Settings()
