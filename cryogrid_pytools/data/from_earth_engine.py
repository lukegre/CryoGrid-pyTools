from functools import lru_cache

import xarray as xr
import ee
import wxee  # noqa
import rioxarray  # noqa

from loguru import logger
from .utils import _decorator_dataarray_to_bbox


@lru_cache(maxsize=1)
def gee_auth():
    """
    Authenticate with Google Earth Engine.

    This function initializes the Earth Engine API. If the user is not
    authenticated, it prompts for authentication and then initializes the API.
    """
    from ee.ee_exception import EEException

    try:
        ee.Initialize()
    except EEException:
        ee.Authenticate()
        ee.Initialize()


@_decorator_dataarray_to_bbox
def get_modis_albedo_500m(bbox_WSEN) -> xr.DataArray:
    """
    Retrieve MODIS albedo data at 500m resolution for a given bounding box.

    Parameters
    ----------
    bbox_WSEN : list or tuple
        Bounding box in the format [west, south, east, north].

    Returns
    -------
    xarray.Dataset
        Dataset containing the MODIS albedo data for the specified region.
    """
    # Initialize Earth Engine
    gee_auth()

    # Load the MODIS image collection
    image_collection = ee.ImageCollection("MODIS/061/MCD43A3")
    logger.info(
        "Fetching MODIS albedo data from Earth Engine (MODIS/061/MCD43A3 @ 500m)"
    )

    # Define the region of interest
    rio = ee.Geometry.BBox(*bbox_WSEN)

    # Filter the image collection for summer months (June to September)
    # in the years 2000-2008 and select the shortwave albedo band
    image_clipped = (
        image_collection.select("Albedo_BSA_shortwave")
        .filterBounds(rio)
        .filterDate("2007-06-01", "2008-09-01")
        .filter(ee.Filter.calendarRange(6, 9, "month"))
    )

    # Compute the 10th percentile of the albedo values
    image_10th_pct = image_clipped.reduce(ee.Reducer.percentile([10])).set(
        "system:time_start", 0
    )

    # Convert the Earth Engine image to an xarray dataset
    ds = (
        image_10th_pct.wx.to_xarray(scale=500, region=rio, progress=False)
        .assign_attrs(period="2000-2008", percentile=10)
        .pipe(lambda x: x * 0.001)  # Scale factor for albedo
        .isel(time=0, drop=True)["Albedo_BSA_shortwave_p10"]
        .rename("modis_albedo_BSA_shortwave")
    )

    return ds


@_decorator_dataarray_to_bbox
def get_aster_ged_100m_v3(bbox_WSEN):
    """
    Retrieve ASTER GED data at 100m resolution for a given bounding box.

    Parameters
    ----------
    bbox_WSEN : list or tuple
        Bounding box in the format [west, south, east, north].

    Returns
    -------
    xarray.Dataset
        Dataset containing ASTER GED data for the specified region.
    """
    # Initialize Earth Engine
    gee_auth()

    # Load the ASTER GED image
    image = ee.Image("NASA/ASTER_GED/AG100_003")
    logger.info(
        "Fetching ASTER-GED emissivity data from Earth Engine (NASA/ASTER_GED/AG100_003 @ 100m)"
    )

    # Define the region of interest
    roi = ee.Geometry.BBox(*bbox_WSEN)

    # Select emissivity bands and elevation
    bands = [f"emissivity_band{b:02d}" for b in range(10, 15)] + ["elevation"]

    # Clip the image to the region of interest
    image_clipped = image.select(bands).clip(roi).set("system:time_start", 0)

    # Convert the Earth Engine image to an xarray dataset
    ds = (
        image_clipped.wx.to_xarray(scale=100, region=roi, progress=False)
        .assign_attrs(period="2000-2008")
        .isel(time=0, drop=True)
    )

    return ds


@_decorator_dataarray_to_bbox
def get_aster_ged_emmis_elev(bbox_WSEN):
    """
    Retrieve ASTER GED emissivity and elevation data for a given bounding box.

    Parameters
    ----------
    bbox_WSEN : list or tuple
        Bounding box in the format [west, south, east, north].

    Returns
    -------
    xarray.Dataset
        Dataset containing emissivity and elevation data for the specified region.
    """
    import xarray as xr

    # Convert bounding box to a tuple for caching
    bbox_WSEN = tuple(bbox_WSEN)

    # Retrieve ASTER GED data
    ds = get_aster_ged_100m_v3(bbox_WSEN)

    # Process emissivity bands
    emissivity_bands = [f"emissivity_band{b:02d}" for b in range(10, 15)]
    emissivity = (
        ds[emissivity_bands]
        .to_array(dim="band")
        .assign_coords(band=range(10, 15))
        .pipe(lambda x: x / 1000)  # Scale factor for emissivity
        # Apply smoothing to reduce noise
        .rolling(x=3, y=3, center=True, min_periods=1)
        .mean()
        .rolling(x=3, y=3, center=True, min_periods=1)
        .mean()
        .assign_attrs(
            long_name="Emissivity",
            units="dimensionless",
            standard_name="emissivity",
            description=(
                "Emissivity in the 8-11 um range "
                "from ASTER Global Emissivity Dataset (ASTER GED)"
            ),
        )
        .assign_coords(
            band_center=xr.DataArray(
                data=[8.3, 8.65, 9.1, 10.6, 11.3],
                dims=("band",),
                attrs=dict(
                    standard_name="band_center",
                    long_name="Band center wavelength",
                    units="um",
                ),
            )
        )
    )

    # Combine elevation and emissivity into a single dataset
    out = ds[["elevation"]].rename(elevation="aster_elevation")
    out["aster_emissivity"] = emissivity

    return out
