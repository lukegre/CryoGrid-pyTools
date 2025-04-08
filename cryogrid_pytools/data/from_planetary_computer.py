try:
    import rioxarray as _rxr  # noqa
    import stackstac as _stackstac  # noqa
    import pystac_client as _pystac_client
    import planetary_computer as _planetary_computer
    from .. import xr_raster_vector as _xrv  # noqa
except ImportError as e:
    missing_package = str(e).split("'")[1]
    raise ImportError(
        f"Missing optional dependency '{missing_package}'. "
        "Please install it using: pip install 'cryogrid_pytools[data]'"
    )

# if cryogrid_pytools[data] dependencies are not installed,
# then warn the user that this package cannot be imported
import xarray as _xr
from loguru import logger as _logger
from . import utils


def search_stac_items_planetary_computer(collection, bbox, **kwargs) -> list:
    """
    Searches for STAC items from the Planetary Computer.

    Parameters
    ----------
    collection : str
        The name of the collection to search within.
    bbox : list
        The bounding box to search within, specified as [minx, miny, maxx, maxy].
    **kwargs : Additional keyword arguments to pass to the search.

    Returns
    -------
    list
        A list of STAC items matching the search criteria.
    """

    URL_PLANETARY_COMPUTER = "https://planetarycomputer.microsoft.com/api/stac/v1"

    catalog = _pystac_client.Client.open(
        url=URL_PLANETARY_COMPUTER, modifier=_planetary_computer.sign_inplace
    )

    search = catalog.search(collections=[collection], bbox=bbox, **kwargs)

    items = search.item_collection()

    return items


@utils._decorator_dataarray_to_bbox
def get_dem_copernicus30(
    bbox_WSEN: list, res_m: int = 30, epsg=32643, smoothing_iters=2, smoothing_size=3
) -> _xr.DataArray:
    """
    Download DEM data from the STAC catalog (default is COP DEM Global 30m).

    Parameters
    ----------
    bbox_WSEN : list
        The bounding box of the area of interest in WSEN format.
    res_m : int
        The resolution of the DEM data in meters.
    epsg : int, optional
        The EPSG code of the projection of the DEM data. Default is
        EPSG:32643 (UTM 43N) for the Pamir region.
    smoothing_iters : int, optional
        The number of iterations to apply the smoothing filter. Default is 2.
        Set to 0 to disable smoothing.
    smoothing_size : int, optional
        The size of the kernel (num pixels) for the smoothing filter. Default is 3.

    Returns
    -------
    xarray.DataArray
        The DEM data as an xarray DataArray with attributes.
    """
    from ..utils import drop_coords_without_dim
    from .utils import check_epsg, smooth_data

    check_epsg(epsg)

    assert res_m >= 30, (
        "The resolution must be greater than 30m for the COP DEM Global 30m dataset."
    )
    res = res_m / 111_111 if epsg == 4326 else res_m

    _logger.info(
        f"Fetching COP DEM Global data from Planetary Computer (cop-dem-glo-30 @ {res_m}m)"
    )
    items = search_stac_items_planetary_computer("cop-dem-glo-30", bbox_WSEN)
    da_dem = _stackstac.stack(
        items=items, bounds_latlon=bbox_WSEN, resolution=res, epsg=epsg
    )

    da_dem = (
        da_dem.mean("time")
        .squeeze()
        .pipe(drop_coords_without_dim)
        .pipe(smooth_data, n_iters=smoothing_iters, kernel_size=smoothing_size)
        .rio.write_crs(f"EPSG:{epsg}")
        .rename("cop_dem_glo_30")
        .assign_attrs(
            source=items[0].links[0].href,  # collection URL
            bbox_request=bbox_WSEN,
        )
    )

    return da_dem


@utils._decorator_dataarray_to_bbox
def get_esa_land_cover(bbox_WSEN: tuple, res_m: int = 30, epsg=32643) -> _xr.DataArray:
    """
    Get the ESA World Cover dataset on the target grid and resolution.

    Parameters
    ----------
    bbox_WSEN : tuple
        Bounding box in the format (West, South, East, North).
    res_m : int, optional
        Resolution in meters. Defaults to 30.
    epsg : int, optional
        EPSG code for the coordinate reference system. Defaults to 32643.

    Returns
    -------
    xr.DataArray
        A DataArray with the land cover data on the target grid. Contains
        attributes 'class_values', 'class_descriptions', 'class_colors' for plotting.
    """
    from .utils import (
        check_epsg,
        get_res_in_proj_units,
    )

    def get_land_cover_classes(item):
        """
        Get the land cover class names, and colors from the ESA World Cover dataset

        Args:
            item (pystac.Item): The STAC item containing the land cover data.

        Returns:
            dict: A dictionary with class values, descriptions, and colors.
        """
        import pandas as pd

        classes = item.assets["map"].extra_fields["classification:classes"]
        df = (
            pd.DataFrame(classes)
            .set_index("value")
            .rename(
                columns=lambda s: s.replace("-", "_")
            )  # bug fix for version 2.7.8 (stacstack back compatibility)
        )

        df["color_hint"] = "#" + df["color_hint"]

        out = dict(
            class_values=df.index.values,
            class_descriptions=df["description"].values,
            class_colors=df["color_hint"].values,
        )

        return out

    # make sure epsg is supported
    check_epsg(epsg)

    # get the units in the projection
    res = get_res_in_proj_units(res_m, epsg, min_res=10)

    _logger.info(
        f"Fetching ESA World Cover (v2.0) data from Planetary Computer (esa-worldcover @ {res_m}m)"
    )
    items = search_stac_items_planetary_computer(
        collection="esa-worldcover",
        bbox=bbox_WSEN,
        query={"esa_worldcover:product_version": {"eq": "2.0.0"}},
    )

    stac_props = dict(
        items=items, assets=["map"], epsg=epsg, bounds_latlon=bbox_WSEN, resolution=res
    )

    da = (
        _stackstac.stack(**stac_props)
        .max(["band", "time"], keep_attrs=True)  # removing the single band dimension
        .rename("esa_world_cover")
        .assign_attrs(**get_land_cover_classes(items[0]))
    )

    return da


@utils._decorator_dataarray_to_bbox
def get_sentinel2_data(
    bbox_WSEN: tuple,
    years=range(2018, 2025),
    assets=["SCL"],
    res_m: int = 30,
    epsg=32643,
    max_cloud_cover=5,
) -> _xr.DataArray:
    """
    Fetches Sentinel-2 data for a given bounding box and time range.

    Parameters
    ----------
    bbox_WSEN : tuple
        Bounding box in the format (West, South, East, North).
    years : range, optional
        Range of years to fetch data for. Defaults to range(2018, 2025).
    assets : list, optional
        List of assets to fetch. Defaults to ['SCL'].
    res_m : int, optional
        Resolution in meters. Defaults to 30.
    epsg : int, optional
        EPSG code for the coordinate reference system. Defaults to 32643.
    max_cloud_cover : int, optional
        Maximum cloud cover percentage. Defaults to 5.

    Returns
    -------
    xr.DataArray
        DataArray containing the fetched Sentinel-2 data.
    """
    from ..utils import drop_coords_without_dim
    from .utils import (
        check_epsg,
        get_res_in_proj_units,
    )

    check_epsg(epsg)
    res = get_res_in_proj_units(res_m, epsg, min_res=10)

    da_list = []
    for year in years:
        _logger.info(
            f"Getting Sentinel-2 SCL granules @{res}m for {year} with max cloud cover = {max_cloud_cover}%"
        )

        t0, t1 = (
            f"{year}-01-01",
            f"{year}-11-15",
        )  # assuming that snow melt is done by mid-November
        items = search_stac_items_planetary_computer(
            collection="sentinel-2-l2a",
            bbox=bbox_WSEN,
            datetime=(t0, t1),
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        )

        da_list += (
            _stackstac.stack(
                items=items,
                assets=assets,
                bounds_latlon=bbox_WSEN,
                resolution=res,
                epsg=epsg,
            ),
        )

    da_granules = _xr.concat(da_list, dim="time")
    da = (
        da_granules.groupby("time")  # granules are not grouped by time
        .max()  # take max value to avoid mixing ints
        .squeeze()  # remove the band dimension
        .where(lambda x: x > 0)
    )  # mask null_values so that pixel coverage can be counted

    da.attrs = {}
    da = da.pipe(drop_coords_without_dim).rio.write_crs(f"EPSG:{epsg}")

    return da


@utils._decorator_dataarray_to_bbox
def get_snow_melt_doy(
    bbox_WSEN: tuple, years=range(2018, 2025), res_m: int = 30, epsg=32643
) -> _xr.DataArray:
    """
    Calculate the snow melt day of year (DOY) from Sentinel-2 SCL data for a given bounding box and years.

    Parameters
    ----------
    bbox_WSEN : tuple
        Bounding box coordinates in the format (West, South, East, North).
    years : range, optional
        Range of years to consider. Defaults to range(2018, 2025).
    res_m : int, optional
        Spatial resolution in meters. Defaults to 30.
    epsg : int, optional
        EPSG code for the coordinate reference system. Defaults to 32643.

    Returns
    -------
    _xr.DataArray
        DataArray containing the snow melt DOY for each year.
    """

    da = get_sentinel2_data(
        bbox_WSEN, years=years, res_m=res_m, epsg=epsg, max_cloud_cover=10
    )

    _logger.info("Calculating snow melt day of year (DOY) from Sentinel-2 SCL data")
    doy = da.groupby("time.year").apply(_calc_sentinel2_snow_melt_doy)

    return doy


def _calc_sentinel2_snow_melt_doy(da_scl) -> _xr.DataArray:
    """
    Calculates the day of year (DOY) when snow melt occurs based on Sentinel-2 SCL data.

    Parameters
    ----------
    da_scl : xarray.DataArray
        The Sentinel-2 SCL data as an xarray DataArray.

    Returns
    -------
    xarray.DataArray
        The day of year when snow melt occurs as an xarray DataArray.
    """

    def drop_poor_coverage_at_end(da, threshold=0.9):
        """
        Drops the time steps at the end of the time series that
        occur after the last point that meets the threshold req.

        Example
        -------
        [0.4, 0.5, 0.3, 0.7, 0.9, 0.3]
        [keep keep keep keep keep drop]
        """
        counts = da.count(["x", "y"]).compute()
        size = da.isel(time=0).size
        frac = counts / size
        good_cover = (
            frac.bfill("time").where(lambda x: x > threshold).dropna("time").time.values
        )
        return da.sel(time=slice(None, good_cover[-1]))

    def find_time_of_lowest_snow_cover(snow_mask, window=10):
        """
        Returns the time step where the snow cover is the lowest
        """
        filled = snow_mask.rolling(time=window, center=True, min_periods=1).max()
        lowest_cover_time = filled.count(["x", "y"]).idxmin()
        return lowest_cover_time

    def get_only_melt_period(snow_mask):
        """
        Drops time steps after snow starts increasing again
        """
        time_snow_cover_min = find_time_of_lowest_snow_cover(snow_mask)
        snow_melt_period = snow_mask.sel(time=slice(None, time_snow_cover_min))
        return snow_melt_period

    def get_max_day_of_year_from_mask(mask):
        """
        Get the maximum day of the year from a given mask.

        Parameters
        ----------
        mask : xarray.DataArray
            A DataArray with a 'time' dimension and boolean values
            indicating the mask.

        Returns
        -------
        xarray.DataArray
            A DataArray containing the maximum day of the year where the mask is True.

        Raises
        ------
        AssertionError
            If 'time' is not a dimension in the mask.
        AssertionError
            If the mask contains data from more than one year.
        """
        assert "time" in mask.dims, "'time' dimension is required"

        years = set(mask.time.dt.year.values.tolist())
        assert len(years) == 1, "Only one year is supported"

        doy_max = (
            mask.time.dt.dayofyear.where(  # get the day of year
                mask
            )  # broadcast the day of year to the mask shape
            .max("time")  # get the last time step
            .astype("float32")
            .rename("day_of_year")
        )

        return doy_max

    scl = da_scl.compute()
    scl_snow_ice = 11

    # only one year allowed
    assert scl.time.dt.year.to_series().unique().size == 1, "Only one year is allowed"

    # find the last time step with good coverage and drop everything after
    # so that we can back fill the snow cover later
    scl_tail_clipped = drop_poor_coverage_at_end(scl, threshold=0.9)
    # mask snow/ice pixels and set values to 1 instead of 11
    snow_mask = scl_tail_clipped.where(lambda x: x == scl_snow_ice) * 0 + 1

    # find the time step where snow cover is the lowest, and remove anything after
    snow_melt = get_only_melt_period(snow_mask)
    # backfill the snow cover (assuming only melt) and create mask
    snow_mask = snow_melt.bfill("time").notnull()
    # compute the melt date based on a mask
    snow_melt_day = get_max_day_of_year_from_mask(snow_mask)

    return snow_melt_day
