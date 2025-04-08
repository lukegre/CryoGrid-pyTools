import xarray as xr
from loguru import logger


def save_raster_3d_to_geotiff(da: xr.DataArray, filename: str, **kwargs):
    """
    Iterates over first dimension of a 3D raster to save each slice as a geotiff

    Parameters
    ----------
    da : xr.DataArray
        A 3D raster
    filename : str
        Output file name template. Can include format specifiers for the
        attributes and coordinates of the DataArray (only works for 1D vars)
    kwargs : dict
        Additional keyword arguments to pass to the `rio.to_raster` method

    Returns
    -------
    None
    """
    import pathlib

    dim = list(da.dims)[0]
    ranges = range(len(da[dim]))

    # get file extension if present in filename
    filename = pathlib.Path(filename)
    ext = filename.suffix

    for i in ranges:
        da_slice = da.isel(**{dim: i})

        possible_keys = dict(**locals(), **da_slice.attrs, **da_slice.coords)
        viable_keys = {}
        for k, v in possible_keys.items():
            if isinstance(v, xr.DataArray):
                if (len(v.dims) <= 1) and (v.size == 1):
                    viable_keys[k] = v.item()
            elif isinstance(v, (int, float, str)):
                viable_keys[k] = v

        fname_stem = filename.stem.format(**viable_keys)

        sname = filename.parent / f"{fname_stem}{ext}"
        da_slice.rio.to_raster(sname, **kwargs)


def get_bounds_latlon(da: xr.DataArray) -> tuple:
    from .vector import bbox_to_geopandas

    bbox = bbox_to_geopandas(da.rio.bounds(), crs=da.rio.crs)
    bbox = bbox.to_crs(4326).total_bounds

    return tuple(bbox)


def prep_raster(da: xr.DataArray, x_axis=-1, y_axis=-2):
    x_name_old = da.dims[x_axis]
    y_name_old = da.dims[y_axis]

    x_name_new = "x"
    y_name_new = "y"

    if x_name_old != x_name_new:
        da = da.rename({x_name_old: x_name_new})
    if y_name_old != y_name_new:
        da = da.rename({y_name_old: y_name_new})

    da = da.sortby(y_name_new, ascending=False)

    if da.rio.crs is None:
        da = _auto_crs(da)

    return da


def _auto_crs(da):
    x = da["x"].values
    y = da["y"].values

    x0, x1 = x.min(), x.max()
    y0, y1 = y.min(), y.max()

    if (x0 >= -180) and (x1 <= 180) and (y0 >= -90) and (y1 <= 90):
        logger.warning(
            "No CRS found in DataArray, assuming EPSG:4326 (lat/lon) since, "
            "coordinates are within the [-180, 180] and [-90, 90] bounds"
        )
        da = da.rio.write_crs("EPSG:4326")
    else:
        logger.warning("No CRS found in DataArray, no default CRS used")

    return da
