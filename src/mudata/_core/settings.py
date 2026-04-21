import inspect
import textwrap
from contextlib import contextmanager
from types import GenericAlias
from typing import Annotated, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class _Settings(BaseSettings):
    """Allows users to customize settings for the mudata package.

    Settings here will generally be for advanced use-cases and should be used with caution.

    For setting an option use :func:`~mudata.settings.override` (local) or set the attributes directly (global)
    i.e., `mudata.settings.my_setting = foo`. For assignment by environment variable, use the variable name in
    all caps with `MUDATA_` as the prefix before import of mudata.
    """

    model_config = SettingsConfigDict(validate_assignment=True, use_attribute_docstrings=True, env_prefix="mudata_")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return init_settings, env_settings, dotenv_settings

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

    @contextmanager
    def override(self, **kwargs):
        """Provides local override via keyword arguments as a context manager."""
        oldsettings = {argname: getattr(self, argname) for argname in kwargs.keys()}
        try:
            for argname, argval in kwargs.items():
                setattr(self, argname, argval)
            yield
        finally:
            for argname, argval in reversed(oldsettings.items()):
                setattr(self, argname, argval)


settings = _Settings()

settings.__doc__ = f"{inspect.getdoc(_Settings)}\n\nThe following options are avaiable:\n"
_Settings.override.__doc__ = f"{inspect.getdoc(_Settings.override)}\n\nParameters\n----------\n"
for fname, field in settings.__class__.model_fields.items():
    type_str = (
        field.annotation.__name__
        if isinstance(field.annotation, type) and not isinstance(field.annotation, GenericAlias)
        else str(field.annotation)
    )
    settings.__doc__ += f"""
.. attribute:: {fname}
   :type: {type_str}
   :value: {field.default!r}

{textwrap.indent(field.description, "   ")}\n"""

    _Settings.override.__doc__ += f"""
{fname} : {type_str}
{textwrap.indent(field.description, "    ")}
    (default `{field.default!r}`)\n"""
