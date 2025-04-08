def enable_xarray_wrapper(func):
    import functools
    import xarray as xr

    """Adds an xarray wrapper for a function without core dimensions."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return xr.apply_ufunc(func, *args, kwargs=kwargs)

    return wrapper


def compute_utm_from_lat_lon(lat, lon):
    epsg = int(32700 - round((45 + lat) / 90, 0) * 100 + round((183 + lon) / 6, 0))
    return epsg
