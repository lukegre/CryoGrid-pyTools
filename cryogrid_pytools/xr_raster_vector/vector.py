import geopandas as gpd
import xarray as xr


def clip_geodata_to_grid(
    df: gpd.GeoDataFrame, target_grid: xr.DataArray
) -> gpd.GeoDataFrame:
    # 1. get crs of target grid
    crs_target = target_grid.rio.crs
    # 2. get bbox of target grid in target crs
    bbox_target = bbox_to_geopandas(target_grid.rio.bounds(), crs=crs_target)

    # 5. reproject shape file to target
    df_target_crs = df.to_crs(crs_target)
    # 6. clip shape file to target bbox
    geometry_target_crs_clipped = df_target_crs.clip_by_rect(*bbox_target.total_bounds)
    df_target_crs["geometry"] = geometry_target_crs_clipped

    return df_target_crs


def bbox_to_geopandas(bbox: tuple, crs="EPSG:4326"):
    """
    Convert a bounding box to a GeoPandas DataFrame with a defined CRS.

    Parameters
    ----------
    bbox : tuple
        A tuple with the bounding box coordinates (west, south, east, north).
    crs : str, optional
        The CRS of the bounding box. The default is 'EPSG:4326'.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoPandas DataFrame with a single row containing the bounding box as a polygon.
        To return a bbox, use df_bbox.total_bounds
    """
    from shapely.geometry import box

    bbox = box(*bbox)
    gdf = gpd.GeoDataFrame(geometry=[bbox], crs=crs)

    return gdf
