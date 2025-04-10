import re


def change_logger_level(level):
    """
    Change the logger level of the cryogrid_pytools logger.

    Parameters
    ----------
    level : str
        Level to change the logger to. Must be one of ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'].
    """
    import sys

    from loguru import logger

    if level in ["INFO", "WARNING", "ERROR", "CRITICAL", "SUCCESS"]:
        format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> - <level>{message}</level>"
    else:
        format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

    logger.remove()
    logger.add(sys.stdout, level=level, format=format)


def drop_coords_without_dim(da):
    """
    Drop coordinates that do not have a corresponding dimension.

    Parameters
    ----------
    da : xarray.DataArray
        The input data array.

    Returns
    -------
    xarray.DataArray
        The data array with dropped coordinates.
    """
    for c in da.coords:
        if c not in da.dims:
            da = da.drop_vars(c)
    return da


def regex_glob(fpath_with_wildcards_or_regex):
    """
    Works like glob.glob but can use regex notation instead.
    Raises error if glob and regex notations are mixed.

    Args:
        fpath_with_wildcards_or_regex (str): Path pattern with glob or regex

    Returns:
        list: Sorted list of matching filenames
    """
    from glob import glob
    from pathlib import Path

    # Regex-specific characters
    regex_indicators = {"+", "|", "(", ")", "{", "}", "$", "^", "\\"}
    has_regex = (
        any(c in fpath_with_wildcards_or_regex for c in regex_indicators)
        or ".*" in fpath_with_wildcards_or_regex
        or ".+" in fpath_with_wildcards_or_regex
    )

    # Glob-specific patterns
    glob_indicators = {"?", "[!"}
    has_glob = any(c in fpath_with_wildcards_or_regex for c in glob_indicators) or (
        "*" in fpath_with_wildcards_or_regex
        and ".*" not in fpath_with_wildcards_or_regex
    )

    # Check for mixed notation
    if has_regex and has_glob:
        raise ValueError(
            "Mixed glob and regex notation detected in the given path. Please use either glob or regex notation, not both."
        )

    # Process according to pattern type
    if has_regex:
        # Handle regex pattern
        path = Path(fpath_with_wildcards_or_regex)
        directory = path.parent if str(path.parent) != "." else Path(".")
        pattern_str = path.name

        try:
            pattern = re.compile(f"^{pattern_str}$")
            matching_files = []

            if directory.exists():
                for item in directory.iterdir():
                    if item.is_file() and pattern.match(item.name):
                        matching_files.append(str(item))

            return sorted(matching_files)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
    else:
        # Use glob for glob patterns or literal paths
        return sorted(glob(fpath_with_wildcards_or_regex))


def check_packages(required_package_names: tuple[str], message: str = None):
    """
    Check if the required packages are installed.

    Parameters
    ----------
    required_package_names : tuple of str
        A tuple of package names to check for installation.
    message : str, optional
        A custom message to display if any package is not installed.

    Raises
    ------
    ImportError
        If any of the required packages are not installed.
    """
    import importlib

    for package_name in required_package_names:
        try:
            importlib.import_module(package_name)
        except ImportError:
            print(package_name)
            raise ImportError(
                message
                or f"Package '{package_name}' is not installed. Please install it."
            )
