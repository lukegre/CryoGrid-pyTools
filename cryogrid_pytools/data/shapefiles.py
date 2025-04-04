import geopandas as gpd
import pooch

from loguru import logger as logger
from memoization import cached as _cached


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

    from .utils import get_earthaccess_session, download_url

    url = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7/regional_files/RGI2000-v7.0-G/RGI2000-v7.0-G-13_central_asia.zip"

    downloader = pooch.HTTPDownloader(
        progressbar=True, headers=get_earthaccess_session().headers
    )
    flist = download_url(url, path=dest_dir, downloader=downloader)

    fname_shp = [f for f in flist if f.endswith(".shp")][0]

    logger.log(
        "INFO",
        "RGI: Fetching Randolph Glacier Inventory - see https://www.glims.org/rgi_user_guide/welcome.html",
    )
    logger.log("DEBUG", f"RGI: URL = {url}")
    logger.log("DEBUG", f"RGI: FILE = {fname_shp}")

    if target_dem is None:
        # reads the whole file
        df = gpd.read_file(fname_shp)
    else:
        # gets the bounding box and then reads the file
        bbox = target_dem.rv.get_bbox_latlon()
        df = gpd.read_file(fname_shp, bbox=bbox).to_crs(target_dem.rio.crs)
        df = df.dissolve()
        ds = df.rv.to_raster(target_dem)
        return ds

    return df


def get_TPRoGI_rock_glaciers(target_dem=None, dest_dir=None) -> gpd.GeoDataFrame:
    """
    Retrieves the TPRoGI dataset and applies a mask for specific countries.

    Returns
    -------
    geopandas.GeoDataFrame
        The GeoDataFrame containing the TPRoGI dataset.

    Examples
    --------
    >>> get_TPRoGI_data()
    GeoDataFrame containing TPRoGI data for Tajikistan and Kyrgyzstan.
    """
    from .utils import read_zenodo_record

    get_flist = [
        "TPRoGI_Extended_Footprint.prj",
        "TPRoGI_Extended_Footprint.dbf",
        "TPRoGI_Extended_Footprint.cpg",
        "TPRoGI_Extended_Footprint.qmd",
        "TPRoGI_Extended_Footprint.shx",
        "TPRoGI_Extended_Footprint.shp",
    ]

    # Download and filter the dataset
    flist = read_zenodo_record(zenodo_id="10732042", flist=get_flist, dest_dir=dest_dir)
    fname = get_shapefile(flist)

    # Apply a mask for Tajikistan and Kyrgyzstan
    mask = get_country_polygons(countries=["Tajikistan"])

    df = gpd.read_file(fname, mask=mask)

    if target_dem is not None:
        bbox = target_dem.rv.get_bbox_latlon()
        df = gpd.read_file(fname, bbox=bbox).to_crs(target_dem.rio.crs)
        df = df.dissolve()
        if len(df) == 0:
            logger.warning("No rock glaciers found in the bounding box.")
            return (target_dem * 0).astype(bool)
        else:
            ds = df.rv.to_raster(target_dem)
        return ds

    return df


def get_country_polygons(countries: list[str] = [], dest_dir=None) -> gpd.GeoDataFrame:
    """
    Retrieves and processes a shapefile containing the boundaries of specified countries.

    Parameters
    ----------
    countries : list of str
        A list of country names for which the shapefile data is to be retrieved.
    dest_dir : str, optional
        The directory where the downloaded shapefile will be stored. If None, the data will
        be stored in the pooch cache directory (~/.cache/pooch/).

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the geometries of the specified countries.

    Notes
    -----
    - The function uses the `pooch` library to download the dataset.
    - The dataset is sourced from OpenDataSoft and is in Parquet format.
    - The "geo_point_2d" column is dropped, and the "geo_shape" column is used as the geometry.

    Examples
    --------
    >>> get_countries_shapefile(["Tajikistan", "Kyrgyzstan"])
    GeoDataFrame containing geometries for Tajikistan and Kyrgyzstan.
    """

    # URL for the world administrative boundaries dataset
    url_world_boundaries = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/world-administrative-boundaries/exports/parquet?lang=en&timezone=Europe%2FBerlin"
    fname = pooch.retrieve(
        url_world_boundaries,
        None,
        fname="world-administrative-boundaries.pq",
        path=dest_dir,
    )

    # Process the dataset and extract geometries for the specified countries
    countries_polygons = (
        gpd.read_parquet(fname)
        .drop(columns=["geo_point_2d"])
        .set_geometry("geo_shape")
        .set_index("name")
    )

    if countries != []:
        countries_polygons = countries_polygons.loc[countries]

    return countries_polygons


def get_shapefile(flist):
    """
    Filters a list of files to find a shapefile.

    Parameters
    ----------
    flist : list of str
        A list of file paths.

    Returns
    -------
    str
        The path to the shapefile.

    Raises
    ------
    ValueError
        If no shapefile is found.

    Examples
    --------
    >>> get_shapefile(["file1.shp", "file2.dbf"])
    'file1.shp'
    """
    fname = [f for f in flist if f.endswith(".shp")]
    if len(fname) == 0:
        raise ValueError("No shapefile found")
    elif len(fname) > 1:
        return fname
    else:
        return fname[0]
