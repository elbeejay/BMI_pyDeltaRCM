import contextlib
import os


def _get_version():
    """Extract version from file.

    Extract version number from single file, and make it availabe everywhere.
    """
    from . import _version
    return _version.__version__()


@contextlib.contextmanager
def temporary_config(_suffix='.yml'):
    """Context manager for a temporary file.

    Creates a temporary file, yields its name, and upon context exit, deletes
    the file.

    Parameters
    ----------
    suffix : :obj:`str`, optional
        Filename extension. Default is '.yml'.

    Yields
    ------
    tmp_yaml : :obj:`str`
        Name of the temporary file.
    """
    import tempfile
    try:
        f = tempfile.NamedTemporaryFile(suffix=_suffix, delete=False)
        tmp_yaml = f.name
        f.close()
        yield tmp_yaml
    finally:
        os.unlink(tmp_yaml)
