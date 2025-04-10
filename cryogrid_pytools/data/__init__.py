from ..utils import check_packages  # noqa

check_packages(
    (
        "bottleneck",
        "earthaccess",
        "ee",
        "era5_downloader",
        "geopandas",
        "ipywidgets",
        "memoization",
        "planetary_computer",
        "pooch",
        "pyarrow",
        "pystac",
        "pyzenodo3",
        "rioxarray",
        "setuptools",
        "stackstac",
        "wxee",
    ),
    message=(
        "You need to install Cryogrid-PyTools with the `data` extra to use this module. \n"
        'Please install it with `pip install "cryogrid-pytools[data]"`. \n'
        'For a full installation use `pip install "cryogrid-pytools[data,viz]".'
    ),
)

from era5_downloader.defaults import (
    create_cryogrid_forcing_fetcher as make_era5_downloader,
)

from .from_earth_engine import get_aster_ged_emmis_elev, get_modis_albedo_500m
from .from_planetary_computer import (
    get_dem_copernicus30,
    get_esa_land_cover,
    get_snow_melt_doy,
)
from .shapefiles import (
    get_country_polygons,
    get_randolph_glacier_inventory,
    get_TPRoGI_rock_glaciers,
)

__all__ = [
    "make_era5_downloader",
    "get_dem_copernicus30",
    "get_esa_land_cover",
    "get_snow_melt_doy",
    "get_modis_albedo_500m",
    "get_aster_ged_emmis_elev",
    "get_TPRoGI_rock_glaciers",
    "get_country_polygons",
    "get_randolph_glacier_inventory",
]
