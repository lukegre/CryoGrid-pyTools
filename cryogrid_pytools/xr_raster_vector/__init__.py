import rioxarray as rxr  # noqa
from loguru import logger  # noqa

from .conversion import (
    polygon_to_raster_bool,
    polygons_to_raster_int,
    raster_bool_to_vector,
    raster_int_to_vector,
)

from . import vector
from . import raster
from . import utils
from . import accessors


def info():
    import xarray as xr
    import numpy as np

    da = xr.DataArray(
        np.ones([1, 1]), dims=["x", "y"], coords={"x": [0], "y": [0]}, name="dummy"
    ).rio.write_crs(4326)
    df = da.to_dataframe()

    accessors_help = ""
    accessors_help += str(df.rv) + "\n"
    accessors_help += str(da.rv) + "\n"
    accessors_help += str(da.morph)

    print("The following accessors have been added:\n\n" + accessors_help)


__all__ = [
    "polygon_to_raster_bool",
    "polygons_to_raster_int",
    "raster_bool_to_vector",
    "raster_int_to_vector",
    "vector",
    "raster",
    "utils",
    "accessors",
    "info",
]
