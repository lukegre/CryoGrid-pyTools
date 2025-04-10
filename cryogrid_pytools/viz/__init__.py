from ..utils import check_packages

check_packages(
    ("folium", "geopandas", "mapclassify", "skimage", "rioxarray"),
    message=(
        "You need to install Cryogrid-PyTools with the `viz` extra to use this module. \n"
        'Please install it with `pip install "cryogrid-pytools[viz]"`.\n'
        'For a full installation use `pip install "cryogrid-pytools[data,viz]".'
    ),
)

import rioxarray as xrx  # noqa
from .. import xr_raster_vector as _xrv  # noqa


from .maps import (
    MARKER_STYLES,
    TILES,
    gridpoints_to_geodataframe,
    finalize_map as finalize_folium_map,
    make_tiles as make_folium_tiles,
    plot_map as plot_folium_map,
)

from .profiles import (
    plot_profile,
    plot_profiles,
)

__all__ = [
    "gridpoints_to_geodataframe",
    "plot_folium_map",
    "make_folium_tiles",
    "finalize_folium_map",
    "TILES",
    "MARKER_STYLES",
    "plot_profile",
    "plot_profiles",
]
