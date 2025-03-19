import numpy as np
import xarray as xr


def calc_profile_props(ground_temperature_profile: xr.DataArray)->xr.Dataset:
    """
    Get properties of the ground temperature profile

    Parameters
    ----------
    ground_temperature_profile : xr.DataArray
        Ground temperature profile in degrees Celsius

    Returns
    -------
    xr.Dataset
        Dataset with the following variables:
        - bottom_thawing_mask: True if thawing from below, False otherwise
        - active_layer_mask: True if active layer, False otherwise
        - permafrost_mask: True if permafrost, False otherwise
        - permafrost_state: 1 if permafrost, 2 if active layer, 3 if bottom thawing
        - active_layer_depth: Depth of the active layer (m)
        - bottom_thawing_depth: Depth of the bottom thawing layer (m)
        - permafrost_thickness: Thickness of the permafrost layer (m)
        - active_layer_temp: Statistics of the active layer temperature
        - permafrost_temp: Statistics of the permafrost temperature

    """
    da = ground_temperature_profile
    depth = da.depth

    assert 'depth' in da.dims, 'depth dimension is required'
    assert 'time' in da.dims, 'time dimension is required'
    assert da.max() < 100, 'temperature should be in degrees Celsius'

    bottom_thawing_mask = detect_bottom_thawing(da)
    # create annual version
    bottom_thawing_mask_year = bottom_thawing_mask.groupby('time.year').max()
    
    # active layer is max thaw depth and shallower
    active_layer_mask_year = detect_active_layer(da)
    # broadcast the years to the time dimension of the profile
    active_layer_mask = active_layer_mask_year.sel(year=da.time.dt.year)
    
    # permamfrost is where not active layer and not bottom thawing
    permafrost_mask = ~active_layer_mask & ~bottom_thawing_mask
    
    ground_thermal_state = (
        permafrost_mask.astype(int) * 1
        + 
        active_layer_mask.astype(int) * 2
        + 
        bottom_thawing_mask.astype(int) * 3
    )
    
    ds = xr.Dataset(attrs=dict(description='Properties of the ground temperature profile'))
    # bollean masks
    ds['ground_temperature'] = da
    ds['ground_thermal_state'] = ground_thermal_state.assign_attrs(
        description='1 if permafrost, 2 if active layer, 3 if bottom thawing',
        values={1: 'permafrost', 2: 'active layer', 3: 'bottom thawing'}
    )

    # derived varaibles
    ds['active_layer_depth'] = get_mask_depth(active_layer_mask_year)
    ds['bottom_thawing_depth'] = get_mask_depth(~bottom_thawing_mask_year).where(lambda x: x != 0)
    ds['permafrost_thickness'] = ds.active_layer_depth - ds.bottom_thawing_depth.fillna(depth.min())

    # layer statistics
    ds['active_layer_temp'] = da.where(active_layer_mask).pipe(get_annual_stats)
    ds['permafrost_temp'] = da.where(permafrost_mask).pipe(get_annual_stats)

    return ds


def detect_upper_thaw_layer(ground_temperature: xr.DataArray)->xr.DataArray:
    """
    Identify the active layer in the ground temperature profile

    The active layer is defined as the layer above the permafrost that thaws
    every year. The active layer is identified as the layer with positive
    temperature that is not thawed from below (see detect_bottom_thawing).

    Parameters
    ----------
    ground_temperature : xr.DataArray
        Ground temperature profile

    Returns
    -------
    xr.DataArray
        True if active layer, False otherwise
    """
    da = ground_temperature.pipe(get_ground_only)

    bottom_thawing = detect_bottom_thawing(ground_temperature)
    active_layer = (da > 0).astype(int) & ~bottom_thawing

    return active_layer


def detect_active_layer(ground_temperature: xr.DataArray)->xr.DataArray:
    """
    Identify the depth of the active layer in the ground temperature profile

    The active layer depth is the maximum thaw depth for each season.
    When no active layer is detected, the depth is set to 0.

    Parameters
    ----------
    ground_temperature : xr.DataArray
        Ground temperature profile
    
    Returns
    -------
    xr.DataArray
        Depth of the active layer (m) reported for each year
    """
    da = ground_temperature.pipe(get_ground_only)

    # get active layer by removing bottom thawing and negative temperatures
    thawed = detect_upper_thaw_layer(da)  # depth x time (hours)
    # detect the maximum temperature in the active layer for each year
    max_temp_annual = thawed.groupby('time.year').max()  # depth x year
    depth = max_temp_annual.depth

    active_layer_depth = (
        depth
        .where(max_temp_annual)  # broadcast depth to thawed_annual
        .pipe(lambda x: x * 0 + 1)
        .ffill('depth')
        .fillna(0)
        .astype(bool)
    )

    return active_layer_depth


def detect_bottom_thawing(ground_temperature: xr.DataArray, min_frozen_frac=0, upper_limit=-5)->xr.DataArray:
    """
    Identify if the saved profile is thawing from below

    This approach approach assumes no thawing if max depth temperature is above 0
    There must also be a frozen layer above the thawing layer

    Parameters
    ----------
    ground_temperature : xr.DataArray
        Ground temperature profile
    min_frozen_frac : float, optional
        Minimum fraction of the profile that must be frozen to be
        considererd thawing from below, by default 0.25
    
    Returns
    -------
    xr.DataArray
        True if thawing from below, False otherwise
    """

    da = ground_temperature.pipe(get_ground_only)

    frozen = (da <= 0).astype(int)
    frozen_limit = frozen.sel(depth=slice(None, upper_limit))
    frac_frozen = frozen_limit.sum('depth') / len(frozen_limit.depth)
    groups = (  # a new group is defined each time the temperature crosses 0 (+ve or -ve)
        frozen 
        .diff(dim='depth')  # compute where the freezing starts and ends
        .pipe(np.abs)  # copmute the absolute value so that thaw is also +1
        .cumsum('depth')
    )
    
    bottom_thawing = (
        (frozen == 0)  # only if the bottom layer is not frozen
        & 
        (groups == 0)  # only if it is the first layer (from the bottom) 
        & 
        (frac_frozen > min_frozen_frac)  # only if a frac of the profile is frozen
        # avoids making warm profiles "thawing from below"
    )
    
    return bottom_thawing


def get_mask_depth(mask: xr.DataArray)->xr.DataArray:
    """
    Finds the depth where mask goes from True to False

    Parameters
    ----------
    mask : xr.DataArray
        Boolean mask with depth as a dimension
    
    Returns
    -------
    xr.DataArray
        Depth in m
    """
    depth = mask.depth

    mask_depth = (
        depth
        .where(mask)  # broadcast depth to mask
        .fillna(0)  # fill with 0 where mask is below depth
        .idxmin('depth')  # get the deepest depth of the mask
        .where(lambda x: x != depth.min())  # if the deepest thawed layer is max depth, then nan
        .fillna(0)  # fill nans with 0
    )

    return mask_depth


def get_ground_only(profile: xr.DataArray)->xr.DataArray:

    da = profile.sortby('depth').sel(depth=slice(-np.inf, 0))
    da = da.dropna('depth', how='all')
    return da


def get_annual_stats(profile, **describe_kwargs):

    def get_group_stats_describe(group):
        from .utils import drop_coords_without_dim

        da = drop_coords_without_dim(group)
        name = da.name

        df = da.to_dataframe()
        
        stats = (
            df[name]
            .describe(**describe_kwargs)
            .to_xarray()
            .drop_sel(index=['count'])
            .rename({'index': 'stat'})
        )
        
        return stats
    
    da = profile.groupby('time.year').apply(get_group_stats_describe)
    return da
