import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import folium 
import munch
from typing import Union


# example of useage: gdf.explore(**shp.GOOGLE_TERRAIN)
TILES = [
    dict(name='Google Terrain', tiles='http://mt0.google.com/vt/lyrs=p&hl=en&x={x}&y={y}&z={z}', attr='Google'),
    dict(name='Google Satellite', tiles='http://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}', attr='Google'),
    dict(name='CartoDB Dark', tiles='cartodbdark_matter'),
    dict(name='CartoDB Light', tiles='cartodbpositron'),
    dict(name='OpenStreetMap', tiles='openstreetmap'),
]

_MARKER_DEFUALTS = dict(marker_type='circle_marker', marker_kwds={'radius': 8}, popup=True)
MARKER_STYLES = munch.Munch(
    red_circle=dict(**_MARKER_DEFUALTS, style_kwds={'fillColor': '#b85451', 'fillOpacity': 1, 'color': 'black'}),
    blue_circle=dict(**_MARKER_DEFUALTS, style_kwds={'fillColor': '#4a66a8', 'fillOpacity': 1, 'color': 'black'}),
)


def finalize_map(m:folium.Map)->folium.Map:
    """
    Finalize a folium map by adding a layer control and fitting the bounds

    Parameters
    ----------
    m : folium.Map
        The map to finalize

    Returns
    -------
    folium.Map
    """

    m.fit_bounds(get_bonuds_from_folium_map(m))
    make_tiles(m)
    folium.LayerControl(collapsed=False).add_to(m)

    return m


def make_tiles(m=None, tiles:list[dict]=TILES):
    """
    Add default tiles to a folium map or create a map if not provided

    Parameters
    ----------
    m : Union[None, folium.Map]
        The map to add the tiles to. If None, a new map is created.
    tiles: list[dict]
        A list of dictionaries with the keys 'name', 'tiles', and 'attr' 
        following the format of folium.TileLayer

    Returns
    -------
    folium.Map
    """

    if m is None:
        m = folium.Map(tiles=None)

    for tile in tiles:
        folium.TileLayer(**tile).add_to(m)

    return m


def plot_map(da:xr.DataArray, m:Union[None, folium.Map]=None, **kwargs)->folium.Map:
    """
    Adds the spatial data to the folium map

    Parameters
    ----------
    da : xr.DataArray
        The data array to add to the map. If the data array is discrete (integers),
        it will be converted to polygons. If the data array is continuous, it will 
        be converted to a raster layer.
    m : Union[None, folium.Map]
        The map to add the data to. If None, a new map is created.
    **kwargs : dict
        Additional keyword arguments to pass to the folium.raster_layers.ImageOverlay
        if da is continuous, or to the df.explore() method if da is discrete.
    
    Returns
    -------
    folium.Map
    """

    low_count = np.unique(da).size < 20

    if low_count:
        df = spatial_discrete_to_polyons(da)
        name = df.columns[-1]
        props = dict(m=m, column=name, name=name, **kwargs)
        if m is None:
            props['m'] = make_tiles()
        m = df.explore(**props)
    else:
        m = spatial_continuous_to_raster(da, m, **kwargs)
        
    return m


def spatial_discrete_to_polyons(da:xr.DataArray)->gpd.GeoDataFrame:
    import numpy as np

    assert da.ndim == 2, "Only 2D data arrays are supported"
    assert np.unique(da).size < 20, "Only discrete data is supported"

    df = da.astype(int).rv.to_polygons().drop(index=0)
    df['class'] = df['class'].astype(str).str.zfill(2)

    name = str(da.name).capitalize().replace('_', ' ')
    df = df.rename(columns={"class": name})

    return df


def spatial_continuous_to_raster(da:xr.DataArray, map:Union[None, folium.Map]=None, **kwargs)->folium.Map:
    """
    Convert a data array that is continuous (not discrete) to a rater layer 
    for a folium map.

    Parameters
    ----------
    da : xr.DataArray
        The data array to convert to a raster layer.
    map : Union[None, folium.Map]
        The map to add the raster layer to. If None, a new map is created.
    **kwargs : dict
        Additional keyword arguments to pass to the folium.raster_layers.ImageOverlay
        constructor.
    """
    from matplotlib import colormaps
    import folium.raster_layers
    
    da = da.astype(float).rio.reproject("EPSG:3857")
    arr = normalize_minmax(da, mask_value=0)
    name = str(da.name).capitalize().replace('_', ' ')

    bounds = calc_bounds(da)

    colormap = colormaps.get_cmap(kwargs.pop('cmap', 'viridis'))

    props = dict(
        image=arr,
        name=name,
        bounds=bounds,
        origin="upper",
        overlay=True,
        show=False,
        mercator_project=True,
        colormap=colormap,
    ) | kwargs
    
    if map is None:
        map = make_tiles()

    folium.raster_layers.ImageOverlay(**props).add_to(map)

    map.fit_bounds(bounds)

    return map


def calc_bounds(da:xr.DataArray)->list:
    """
    Computes the lat/lon bounds of a dataset

    Parameters
    ----------
    da : xr.DataArray
        The data array to compute the bounds for that has an 
        assigned crs.
    
    Returns
    -------
    list
        The bounds of the dataset in the form of [[lat0, lon0], [lat1, lon1]]
    """

    assert da.rio.crs is not None, "The dataset must have a crs assigned"

    da_latlon = da.rio.reproject("EPSG:4326")
    bounds = [
        [da_latlon.y.min().item(), da_latlon.x.min().item()],
        [da_latlon.y.max().item(), da_latlon.x.max().item()],
    ]
    return bounds


def get_bonuds_from_folium_map(m:folium.Map)->list[list[float]]:
    import folium.features

    for name in m._children:
        child = m._children[name]
        if isinstance(child, folium.features.GeoJson):
            data = child.data
            w, s, e, n = data['bbox']
            w, s, e, n = [float(a) for a in [w, s, e, n]]
            bounds = [[s, w], [n, e]]
            return bounds
        
    return m.get_bounds()


def gridpoints_to_geodataframe(ds_flat:xr.Dataset, lat_name='lat', lon_name='lon')->gpd.GeoDataFrame:
    """
    Converts a flat (tabular) xarray dataset to a geopandas dataframe. 
    Designed to use for folium plotting (df.explore())

    Parameters
    ----------
    ds_flat : xr.Dataset
        The flat dataset to convert to a geodataframe with a 
        signle dimension that is also the dimension for the latitude
        and longitude.
    lat_name : str
        The name of the latitude coordinate.
    lon_name : str
        The name of the longitude coordinate.

    Returns
    -------
    gpd.GeoDataFrame
        The geodataframe with the lat, lon coordinates as the geometry
        column. The remaining variables in the input Dataset are converted
        to columns in the output GeoDataFrame. When plotting these data 
        with df.explore, the additional columns will be shown in the 
        tooltip and popup. 
    """
    
    latlon = [lat_name, lon_name]
    variables = list(ds_flat.data_vars.keys())
    columns = list(set(variables) - set(latlon))
    columns = sorted(columns)

    df = ds_flat[latlon + columns].to_dataframe()
    data = pd.DataFrame(df.drop(columns=latlon))
    
    gdf = gpd.GeoDataFrame(
        data, 
        geometry=gpd.points_from_xy(df.lon, df.lat), 
        crs='EPSG:4326').round(2)
    
    return gdf


def normalize_minmax(da: xr.DataArray, mask_value=0)->np.ndarray:
    da_norm = (da - da.min()) / (da.max() - da.min())
    da_masked = da_norm.where(da != mask_value)
    arr = da_masked.values
    return arr


def _marker_with_image_popup(ser:gpd.GeoSeries, **kwargs):

    drop_cols = ['image', 'geometry']
    tootlip_classes = 'table table-striped table-hover table-condensed table-responsive'
    tooltip = ser.drop(drop_cols, errors='ignore').to_frame().to_html(classes=tootlip_classes)
    
    popup = folium.Popup(ser.image) if 'image' in ser else tooltip

    x, y = ser.geometry.x, ser.geometry.y
    marker = folium.CircleMarker(
        (y, x),
        popup=popup,
        tooltip=tooltip,
        **kwargs)
    
    return marker


def plot_geodataframe_with_image_popups(df: gpd.GeoDataFrame, m=None, **kwargs):
    """
    Plot a GeoDataFrame with image popups

    Parameters
    ----------
    df : gpd.GeoDataFrame
        The GeoDataFrame to plot with image popups
    **kwargs : dict
        Additional keyword arguments to pass to the folium.CircleMarker
        constructor.
    """
    import folium

    if m is None:
        m = make_tiles()

    feature_group = folium.FeatureGroup(name=kwargs.pop('name', 'Markers'))
    for idx, row in df.iterrows():
        marker = _marker_with_image_popup(row, **kwargs)
        marker.add_to(feature_group)
    feature_group.add_to(m)

    return m