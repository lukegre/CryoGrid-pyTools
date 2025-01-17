try:
    import earthaccess as _earthaccess
    import geopandas as _gpd
    import planetary_computer as _planetary_computer
    import pooch as _pooch
    import pystac_client as _pystac_client
    import stackstac as _stackstac
    import rioxarray as _rxr
    import xarray_raster_vector as _xrv
    from memoization import cached as _cached
    from era5_downloader.defaults import create_cryogrid_forcing_fetcher as make_era5_downloader
except ImportError as e:
    missing_package = str(e).split("'")[1]
    raise ImportError(
        f"Missing optional dependency '{missing_package}'. "
        "Please install it using: pip install 'cryogrid_pytools[data]'"
    )

# if cryogrid_pytools[data] dependencies are not installed, 
# then warn the user that this package cannot be imported
from loguru import logger as _logger
import xarray as _xr


def _decorator_dataarray_to_bbox(func):
    """
    A decorator that processes the first argument of the decorated function to handle 
    either an xarray DataArray or a bounding box tuple. If the first argument is a 
    DataArray, it extracts the bounding box from the DataArray and reprojects the 
    output to match the DataArray's projection.
    
    Parameters
    ----------
    func : callable
        The function to be decorated. The function should accept a 
        bounding box as its first argument.
    
    Returns
    -------
    callable
        The wrapped function with the additional functionality of handling 
        DataArray and reprojecting the output.
    """
    @_cached
    def wrapper(*args, **kwargs):
        
        if len(args) >= 1:
            bbox_or_target = args[0]
            args = args[1:]
        else:
            bbox_or_target = kwargs.pop('bbox', None) or kwargs.pop('bbox_WSEN', None)
    
        if isinstance(bbox_or_target, _xr.DataArray):
            da = bbox_or_target
            bbox = da.rv.get_bbox_latlon()
            out = func(bbox, *args, **kwargs)
            out = out.rio.reproject_match(da)

        elif isinstance(bbox_or_target, tuple):
            bbox =  bbox_or_target
            out = func(bbox_or_target, *args, **kwargs)

        return out
    
    return wrapper


@_decorator_dataarray_to_bbox
def get_dem_copernicus30(bbox_WSEN:list, res_m:int=30, epsg=32643, smoothing_iters=2, smoothing_size=3)->_xr.DataArray:
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

    check_epsg(epsg)

    assert res_m >= 30, "The resolution must be greater than 30m for the COP DEM Global 30m dataset."
    res = res_m / 111111 if epsg == 4326 else res_m

    _logger.info("Fetching COP DEM Global 30m data from Planetary Computer")
    items = search_stac_items_planetary_computer('cop-dem-glo-30', bbox_WSEN)
    da_dem = _stackstac.stack(
        items=items, 
        bounds_latlon=bbox_WSEN,
        resolution=res,
        epsg=epsg)
    
    da_dem = (
        da_dem
        .mean('time')
        .squeeze()
        .pipe(drop_coords_without_dim)
        .pipe(smooth_data, n_iters=smoothing_iters, kernel_size=smoothing_size)
        .rio.write_crs(f"EPSG:{epsg}")
        .assign_attrs(
            source=items[0].links[0].href,  # collection URL
            bbox_request=bbox_WSEN))

    return da_dem


@_decorator_dataarray_to_bbox
def get_esa_land_cover(bbox_WSEN:tuple, res_m:int=30, epsg=32643)->_xr.DataArray:
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

    def get_land_cover_classes(item):
        """ 
        Get the land cover class names, and colors from the ESA World Cover dataset

        Args:
            item (pystac.Item): The STAC item containing the land cover data.

        Returns:
            dict: A dictionary with class values, descriptions, and colors.
        """
        import pandas as pd

        classes = item.assets['map'].extra_fields['classification:classes']
        df = pd.DataFrame(classes).set_index('value')

        df['color-hint'] = '#' + df['color-hint']

        out = dict(
            class_values = df.index.values,
            class_descriptions = df['description'].values,
            class_colors = df['color-hint'].values)

        return out

    # make sure epsg is supported
    check_epsg(epsg)

    # get the units in the projection
    res = get_res_in_proj_units(res_m, epsg, min_res=10)

    _logger.info("Fetching ESA World Cover (v2.0) data from Planetary Computer")
    items = search_stac_items_planetary_computer(
        collection='esa-worldcover', 
        bbox=bbox_WSEN, 
        query={'esa_worldcover:product_version': {'eq': '2.0.0'}})

    stac_props = dict(
        items=items, 
        assets=['map'],
        epsg=epsg,
        bounds_latlon=bbox_WSEN,
        resolution=res)

    da = (
        _stackstac.stack(**stac_props)
        .max(['band', 'time'], keep_attrs=True)  # removing the single band dimension
        .rename('land_cover')
        .assign_attrs(**get_land_cover_classes(items[0]))
    )
    
    return da


@_decorator_dataarray_to_bbox
def get_sentinel2_data(bbox_WSEN:tuple, years=range(2018, 2025), assets=['SCL'], res_m:int=30, epsg=32643, max_cloud_cover=5)->_xr.DataArray:
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
    
    check_epsg(epsg)
    res = get_res_in_proj_units(res_m, epsg, min_res=10)

    da_list = []
    for year in years:
        _logger.info(f"Getting Sentinel-2 SCL granules @{res}m for {year} with max cloud cover = {max_cloud_cover}%")
    
        t0, t1 = f"{year}-01-01", f"{year}-11-15"  # assuming that snow melt is done by mid-November
        items = search_stac_items_planetary_computer(
            collection='sentinel-2-l2a',
            bbox=bbox_WSEN,
            datetime=(t0, t1),
            query={'eo:cloud_cover': {'lt': max_cloud_cover}})
        
        da_list += _stackstac.stack(
            items=items, 
            assets=assets,
            bounds_latlon=bbox_WSEN,
            resolution=res,
            epsg=epsg),
    
    da_granules = _xr.concat(da_list, dim='time')
    da = (
        da_granules  # granules are not grouped by time
        .groupby('time').max()  # take max value to avoid mixing ints
        .squeeze()  # remove the band dimension
        .where(lambda x: x > 0))  # mask null_values so that pixel coverage can be counted
    
    da.attrs = {}
    da = da.pipe(drop_coords_without_dim).rio.write_crs(f"EPSG:{epsg}")

    return da


@_decorator_dataarray_to_bbox
def get_snow_melt_doy(bbox_WSEN:tuple, years=range(2018, 2025), res_m:int=30, epsg=32643)->_xr.DataArray:
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
    
    da = get_sentinel2_data(bbox_WSEN, years=years, res_m=res_m, epsg=epsg, max_cloud_cover=10)

    _logger.info("Calculating snow melt day of year (DOY) from Sentinel-2 SCL data")
    doy = da.groupby('time.year').apply(calc_sentinel2_snow_melt_doy)

    return doy
    

@_cached
def get_randolph_glacier_inventory(target_dem=None, dest_dir=None):
    """
    Fetches the Randolph Glacier Inventory (RGI) data and returns it as a GeoDataFrame or raster dataset.

    Parameters
    ----------
    target_dem : optional
        A digital elevation model (DEM) object. If provided, the function will return 
        the RGI data clipped to the bounding box of the DEM and reprojected to the DEM's CRS.
    dest_dir : str, optional
        The directory where the downloaded RGI data will be stored. If None, the data will
        be stored in the pooch cache directory (~/.cache/pooch/).

    Returns
    -------
    GeoDataFrame or raster dataset
        If target_dem is None, returns a GeoDataFrame containing the RGI data.
        If target_dem is provided, returns a raster dataset clipped and reprojected to the DEM.
    """

    url = 'https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7/regional_files/RGI2000-v7.0-G/RGI2000-v7.0-G-13_central_asia.zip'

    downloader = _pooch.HTTPDownloader(progressbar=True, headers=get_earthaccess_session().headers)
    flist = download_url(url, path=dest_dir, downloader=downloader)

    fname_shp = [f for f in flist if f.endswith('.shp')][0]

    _logger.log("INFO", f"RGI: Fetching Randolph Glacier Inventory - see https://www.glims.org/rgi_user_guide/welcome.html")
    _logger.log("DEBUG", f"RGI: URL = {url}")
    _logger.log("DEBUG", f"RGI: FILE = {fname_shp}")

    if target_dem is None:
        # reads the whole file
        df = _gpd.read_file(fname_shp)
    else:
        # gets the bounding box and then reads the file
        bbox = target_dem.rv.get_bbox_latlon()
        df = _gpd.read_file(fname_shp, bbox=bbox).to_crs(target_dem.rio.crs)
        df = df.dissolve()
        ds = df.rv.to_raster(target_dem)
        return ds

    return df


def smooth_data(da: _xr.DataArray, kernel_size:int=3, n_iters=1)->_xr.DataArray:
    """
    Smooth the data using a rolling mean filter (box kernel).

    Parameters
    ----------
    da : xarray.DataArray
        The input data as an xarray DataArray.
    kernel_size : int
        The size of the kernel for the rolling mean filter.
    n_iters : int
        The number of iterations to apply the filter.

    Returns
    -------
    xarray.DataArray
        The smoothed data as an xarray DataArray.
    """

    da_smooth = da.copy()
    for _ in range(n_iters):
        da_smooth = da_smooth.rolling(
            x=kernel_size, y=kernel_size, center=True, min_periods=1).mean()

    if n_iters:
        da_smooth = da_smooth.assign_attrs(
            smoothing_kernel='box_kernel',
            smoothing_kernel_size=kernel_size,
            smoothing_iterations=n_iters)
    
    return da_smooth


def calc_sentinel2_snow_melt_doy(da_scl)-> _xr.DataArray:
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
        counts = da.count(['x', 'y']).compute()
        size = da.isel(time=0).size
        frac = counts / size
        good_cover = (
            frac
            .bfill('time')
            .where(lambda x: x > threshold)
            .dropna('time')
            .time.values)
        return da.sel(time=slice(None, good_cover[-1]))

    def find_time_of_lowest_snow_cover(snow_mask, window=10):
        """
        Returns the time step where the snow cover is the lowest
        """
        filled = snow_mask.rolling(time=window, center=True, min_periods=1).max()
        lowest_cover_time = filled.count(['x', 'y']).idxmin()
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
        assert 'time' in mask.dims, "'time' dimension is required"

        years = set(mask.time.dt.year.values.tolist())
        assert len(years) == 1, "Only one year is supported"

        doy_max = (
            mask.time.dt.dayofyear  # get the day of year
            .where(mask)  # broadcast the day of year to the mask shape
            .max('time')  # get the last time step 
            .astype('float32')
            .rename('day_of_year'))
        
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
    snow_mask = snow_melt.bfill('time').notnull()
    # compute the melt date based on a mask
    snow_melt_day = get_max_day_of_year_from_mask(snow_mask)
    
    return snow_melt_day


def get_earthaccess_session():
    """
    Logs into earthaccess and gets session info.
    
    Returns
    -------
    session
        The session information.
    """

    auth = _earthaccess.login(persist=True)
    session = auth.get_session()

    return session


def download_url(url, **kwargs):
    """
    Download a file from a given URL and process it if necessary.

    Parameters
    ----------
    url : str
        The URL of the file to download.
    **kwargs : Additional properties passed to pooch.retrieve. 
        These properties will override the default settings.

    Returns
    -------
    list
        A list of file paths to the downloaded (and possibly 
        decompressed) files.
    """
    
    if url.endswith('.zip'):
        processor = _pooch.Unzip()
    elif url.endswith('.tar.gz'):
        processor = _pooch.Untar()
    else:
        processor = None

    default_props = dict(
        known_hash=None,
        fname=url.split('/')[-1],
        path=None,
        downloader=_pooch.HTTPDownloader(progressbar=True),
        processor=processor)

    props = default_props | kwargs
    flist = _pooch.retrieve(url=url, **props)

    return flist


def search_stac_items_planetary_computer(collection, bbox, **kwargs)->list:
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
        url=URL_PLANETARY_COMPUTER,
        modifier=_planetary_computer.sign_inplace)

    search = catalog.search(
        collections=[collection],
        bbox=bbox, 
        **kwargs)

    items = search.item_collection()

    return items


def check_epsg(epsg:int)->bool:
    """
    Check if the provided EPSG code is valid.

    Parameters
    ----------
    epsg : int
        The EPSG code to be checked.

    Returns
    -------
    bool
        True if the EPSG code is valid, False otherwise.

    Raises
    ------
    AssertionError
        If the EPSG code is not valid.
    """
    def check_epsg(epsg: int) -> bool:
        """
        Check if the provided EPSG code is valid.

        This function checks whether the given EPSG code is either a UTM code 
        (starting with '326') or the Lat/Lon code (4326).

        Args:
            epsg (int): The EPSG code to be checked.

        Returns:
            bool: True if the EPSG code is valid, False otherwise.

        Raises:
            AssertionError: If the EPSG code is not valid.
        """
    is_valid_epsg = str(epsg).startswith('326') or (epsg == 4326)
    assert is_valid_epsg, "The EPSG code must be UTM (326xx) or Lat/Lon (4326)."


def get_res_in_proj_units(res_m, epsg, min_res=30):
    """
    Check if the resolution is valid for the given EPSG code.

    Parameters
    ----------
    res_m : int
        The resolution in meters.
    epsg : int
        The EPSG code of the projection.
    min_res : int
        The minimum resolution required for the dataset.

    Returns
    -------
    res : int
        The resolution in the units of the projection.
    """
    message = f"The resolution must be greater than {min_res}m for this collection"
    assert res_m > min_res, message
    res = res_m / 111111 if epsg == 4326 else res_m

    return res


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
