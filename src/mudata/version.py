"""Compute the version number and store it in the `__version__` variable.

Based on <https://github.com/maresb/hatch-vcs-footgun-example>.
"""


def _get_hatch_version():
    """Compute the most up-to-date version number in a development environment.

    Returns `None` if Hatchling is not installed, e.g. in a production environment.

    For more details, see <https://github.com/maresb/hatch-vcs-footgun-example/>.
    """
    import os

    try:
        from hatchling.metadata.core import ProjectMetadata
        from hatchling.plugin.manager import PluginManager
        from hatchling.utils.fs import locate_file
    except ImportError:
        # Hatchling is not installed, so probably we are not in
        # a development environment.
        return None

    pyproject_toml = locate_file(__file__, "pyproject.toml")
    if pyproject_toml is None:
        return None
    root = os.path.dirname(pyproject_toml)
    metadata = ProjectMetadata(root=root, plugin_manager=PluginManager())
    # Version can be either statically set in pyproject.toml or computed dynamically:
    return metadata.core.version or metadata.hatch.version.cached


__version__ = _get_hatch_version()
__version_tuple__ = None
if not __version__:  # not in development mode
    from ._version import __version__
