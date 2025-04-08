import geopandas as gpd
import numpy as np
import xarray as xr
import pandas as pd
from loguru import logger
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*Geometry is in a geographic CRS.*"
)


def raster_bool_to_vector(
    da: xr.DataArray, combine_polygons=False, buffer_dist=0, simplify_dist=0
) -> gpd.GeoDataFrame:
    """
    Converts a rasterized mask to a vectorized representation.

    Parameters
    ----------
    da : xr.DataArray (bool)
        The rasterized mask to convert to a polygon
    combine_polygons : bool, optional
        If True, all polygons are combined into a single polygon. if False, each
        connected component is a separate polygon. The default is False.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame with the vectorized representation of the mask.
    """
    import rasterio.features
    from shapely import geometry

    assert da.dtype == bool, "Input array must be boolean"
    assert da.ndim == 2, "Input array must be 2D"
    transform = da.rio.transform()

    arr = da.values.astype(np.uint8)
    shapes = rasterio.features.shapes(arr, transform=transform)
    crs = da.rio.crs

    if crs is None:
        logger.warning("No CRS found in DataArray, assuming EPSG:4326 (lat/lon)")
        crs = "EPSG:4326"

    def get_coord(s):
        geometry.Polygon(s[0]["coordinates"][0])

    polygons = [get_coord(shape) for shape in shapes if shape[1] == 1]

    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    if combine_polygons:
        gdf = gdf.dissolve()

    if buffer_dist > 0:
        gdf["geometry"] = gdf.buffer(buffer_dist).buffer(-buffer_dist)
    if simplify_dist > 0:
        gdf["geometry"] = gdf.simplify(simplify_dist)

    gdf = gdf.set_crs(crs)

    return gdf


def raster_int_to_vector(
    da: xr.DataArray, names=None, buffer_dist=0, simplify_dist=0
) -> gpd.GeoDataFrame:
    """
    Converts a rasterized mask with several classes to a vectorized representation.

    Parameters
    ----------
    da : xr.DataArray (int)
        The rasterized mask with several classes.
    names : list, optional
        The names of the classes. The default is None, in which case the classes are numbered.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame with the vectorized representation of the mask.
    """

    assert da.dtype == int, "Input array must be integer"

    mask_values = np.sort(np.unique(da.values))
    n_classes = mask_values.size

    if n_classes > 20:
        raise ValueError("Too many classes to convert to vector")

    if names is None:
        names = [str(i) for i in mask_values]
    else:
        assert len(names) == n_classes, (
            f"Number of names (n={len(names)}) must match number of classes (n={n_classes})"
        )

    polygons = []
    for m, name in zip(mask_values, names):
        # logger.debug(f"Converting class {name} [{m}] to vector")
        mask = da == m
        polygons += (
            raster_bool_to_vector(
                da=mask,
                combine_polygons=True,
                simplify_dist=simplify_dist,
                buffer_dist=buffer_dist,
            ),
        )

    polygons = pd.concat(polygons, ignore_index=True)
    polygons["class"] = names

    return polygons


def polygon_to_raster_bool(polygon, da_target):
    """
    Convert a Shapely polygon to a binary mask that matches the grid of an xarray.DataArray.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon or geopandas.GeoDataFrame
        The polygon to convert to a raster mask.
    da_target : xr.DataArray
        The target grid to match the mask to (spatial dimensions must be 'x' and 'y').

    Returns
    -------
    mask_da : xr.DataArray
        A boolean DataArray with the mask on the target grid.
    """
    import rasterio
    from rasterio.features import rasterize
    from shapely.geometry import mapping

    if isinstance(polygon, (gpd.GeoSeries, gpd.GeoDataFrame)):
        polygon = polygon.unary_union

    # Get the spatial dimensions of the data array
    if "x" in da_target.dims and "y" in da_target.dims:
        x, y = "x", "y"
    else:
        raise ValueError("Data array must have 'x' and 'y' dimensions")

    # make sure lat is descending, otherwise upside down coords
    da_target = da_target.sortby("y", ascending=False)

    # Define the transformation from pixel coordinates to geographical coordinates
    transform = rasterio.transform.from_bounds(
        min(da_target[x].values),
        min(da_target[y].values),
        max(da_target[x].values),
        max(da_target[y].values),
        len(da_target[x]),
        len(da_target[y]),
    )

    # Rasterize the polygon
    mask = rasterize(
        [mapping(polygon)],
        out_shape=(len(da_target[y]), len(da_target[x])),
        transform=transform,
        fill=0,
        out=None,
        all_touched=True,
        dtype=np.uint8,
    )

    # Create a DataArray from the mask
    mask_da = xr.DataArray(
        mask, dims=(y, x), coords={y: da_target[y], x: da_target[x]}
    ).astype(bool)

    return mask_da


def polygons_to_raster_int(
    df: gpd.GeoDataFrame, da_target: xr.DataArray, by_column=None, **joblib_kwargs
) -> xr.DataArray:
    """
    Convert a GeoDataFrame with polygons to a raster mask with integer values.

    Each row of polygons in the GeoDataFrame is converted to a separate integer
    value in the raster mask. The integer values are assigned in the order of the
    rows in the GeoDataFrame. The conversion is run in parallel using joblib.

    Parameters
    ----------
    df : gpd.GeoDataFrame
        The GeoDataFrame with polygons to convert to a raster mask.
    da_target : xr.DataArray
        The target grid to match the mask to (spatial dimensions must be 'x' and 'y').
    by_column : str, optional
        The column in the GeoDataFrame to group the polygons by. If None, then each
        row is converted to a separate integer value. The default is None.
    joblib_kwargs : dict, optional
        Additional keyword arguments to pass to joblib.Parallel. The default is {}.

    Returns
    -------
    xr.DataArray
        A DataArray with the raster mask with integer values.
    """
    import joblib

    if by_column is not None:
        assert by_column in df.columns, f"Column {by_column} not found in DataFrame"
        df = df.dissolve(by=by_column).reset_index()

    func = joblib.delayed(
        lambda ser, i: polygon_to_raster_bool(ser, da_target) * (i + 1)
    )
    tasks = [func(row.geometry, i) for i, row in df.iterrows()]

    props = dict(n_jobs=-1, backend="threading")
    props.update(joblib_kwargs)

    polygons = joblib.Parallel(**props)(tasks)

    polygons = (
        xr.concat(polygons, dim="polygons")
        .assign_coords(polygons=df.index)
        .max(dim="polygons")
        .astype(int)
    )

    return polygons
