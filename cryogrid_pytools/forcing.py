# depends on matlab_helpers.py
import xarray as xr


def _era5_mat_dict_to_xarray(era5_dict: dict)->xr.Dataset:
    """
    Convert a dictionary with the ERA5 forcing variables to a xarray Dataset

    Parameters
    ----------
    era5_dict : dict
        Dictionary with the ERA5 forcing variables. Must contain keys: 
            lon, lat, p, t, dims, 
            u10, v10, u, v, Td2, T2, T, Z, q, P, ps, SW, LW, S_TOA, Zs,
            wind_sf, q_sf, ps_sf, rad_sf, T_sf, P_sf

    Returns
    -------
    xr.Dataset
        Dataset with the ERA5 forcing variables and variables are scaled to the original units
    """
    dat = era5_dict

    out = xr.Dataset(
        coords=dict(
            lon=dat['lon'],
            lat=dat['lat'],
            level=dat['p'],
            time=dat['t']))

    for key in dat:
        if key in ['lon', 'lat', 'p', 't', 'dims']:
            continue
        
        arr = dat[key]
        ndim = len(arr.shape)

        if ndim == 0:
            dims = []
        elif ndim == 2:
            dims = ['lon', 'lat']
        elif ndim == 3:
            dims = ['lon', 'lat', 'time']
        elif ndim == 4:
            dims = ['lon', 'lat', 'level', 'time']
        else:
            raise ValueError(f'Unexpected number of dimensions for `{key}`: {ndim}')
        
        out[key] = xr.DataArray(arr, dims=dims)

    out['u10'] = out['u10'] * dat['wind_sf']
    out['v10'] = out['v10'] * dat['wind_sf']
    out['u'] = out['u'] * dat['wind_sf']
    out['v'] = out['v'] * dat['wind_sf']

    out['Td2'] = out['Td2'] * dat['T_sf']
    out['T2'] = out['T2'] * dat['T_sf']
    out['T'] = out['T'] * dat['T_sf']

    out['SW'] = out['SW'] * dat['rad_sf']
    out['LW'] = out['LW'] * dat['rad_sf']
    out['S_TOA'] = out['S_TOA'] * dat['rad_sf']
    
    out['ps'] = out['ps'] * dat['ps_sf']
    out['P'] = out['P'] * dat['P_sf']
    out['q'] = out['q'] * dat['q_sf']

    out['Zs'] = out['Zs'].astype(float)
    out['Z'] = out['Z'].astype(float)

    out = out.transpose('time', 'level', 'lat', 'lon')

    return out


def read_mat_ear5(filename: str)->xr.Dataset:
    """
    Read the ERA5.mat forcing file for CryoGrid and return a xarray Dataset

    Parameters
    ----------
    filename : str
        Path to the ERA5.mat file

    Returns
    -------
    xr.Dataset
        Dataset with the variables from the ERA5.mat file
    """
    import pathlib
    from .matlab_helpers import read_mat_struct_flat_as_dict

    filename = pathlib.Path(filename).expanduser().absolute().resolve()

    dat = read_mat_struct_flat_as_dict(filename)
    out = _era5_mat_dict_to_xarray(dat)

    out = out.assign_attrs(
        info=(
            'Data read in from CryoGrid ERA5 forcing file. '
            'Data has been scaled to the original units. '
            'Data has been transposed from [lon, lat, level, time] --> [time, level, lat, lon]. '
            'See the ERA5 documentation for more info about the units etc.'),
        source=filename,
    )

    return out


def era5_to_matlab(ds: xr.Dataset)->dict:
    """
    Converrt a merged netCDF file from the Copernicus CDS to 
    a dictionary that matches the expected format of 
    the CryoGrid.POST_PROC.read_mat_ERA class (in MATLAB)

    Parameters
    ----------
    ds : xr.Dataset
        Dataset from the ERA5 Copernicus CDS with variables required for 
        the CryoGrid.POST_PROC.read_mat_ERA class 
        single_levels   = [u10, v10, sp, d2m, t2m, ssrd, strd, tisr, tp, Zs]
        pressure_levels = [t, z, q, u, v]
        Note that Zs in the single levels is a special case since it is only 
        downloaded for a single date at the surface (doesn't change over time)

    Returns
    -------
    dict
        Dictionary with the variables mapped to names that are expected by 
        CryoGrid.POST_PROC.read_mat_ERA
    """
    from .matlab_helpers import datetime2matlab
    import numpy as np

    # transpose to lon x lat x time (original is time x lat x lon)
    ds = ds.transpose('longitude', 'latitude', 'level', 'time')

    era = dict()
    era['dims'] = 'lon x lat (x pressure_levels) x time'
    # while lat and lon have to be [coord x 1]
    era['lat'] = ds['latitude'].values[:, None]
    era['lon'] = ds['longitude'].values[:, None]
    # pressure levels have to be [1 x coord] - only when pressure_levels present
    era['p'] = ds['level'].values[None] * 100 
    # time for some reason has to be [1 x coord]
    era['t'] = datetime2matlab(ds.time)[None] 
    # geopotential height at surface
    era['Zs'] = ds.Zs.values / 9.81  # gravity m/s2

    # single_level variables
    # wind and pressure (no transformations)
    era['u10'] = ds['u10'].values
    era['v10'] = ds['v10'].values
    era['ps'] = ds['sp'].values
    # temperature variables (degK -> degC)
    era['Td2'] = ds['d2m'].values - 273.15
    era['T2'] = ds['t2m'].values - 273.15
    # radiation variables (/sec -> /hour)
    era['SW'] = ds['ssrd'].values / 3600
    era['LW'] = ds['strd'].values / 3600
    era['S_TOA'] = ds['tisr'].values / 3600
    # precipitation (m -> mm)
    era['P'] = ds['tp'].values * 1000

    # pressure levels
    era['T'] = ds['t'].values - 273.15  # K to C
    era['Z'] = ds['z'].values / 9.81  # gravity m/s2
    era['q'] = ds['q'].values
    era['u'] = ds['u'].values
    era['v'] = ds['v'].values

    # scaling factors
    era['wind_sf'] = 1e-2
    era['q_sf'] = 1e-6
    era['ps_sf'] = 1e2
    era['rad_sf'] = 1e-1
    era['T_sf'] = 1e-2
    era['P_sf'] = 1e-2

    # apply scaling factors (done in the original, so we do it here)
    # wind scaling
    era['u']     = (era['u']     / era['wind_sf']).astype(np.int16)
    era['v']     = (era['v']     / era['wind_sf']).astype(np.int16)
    era['u10']   = (era['u10']   / era['wind_sf']).astype(np.int16)
    era['v10']   = (era['v10']   / era['wind_sf']).astype(np.int16)
    # temperature scaling
    era['T']     = (era['T']     / era['T_sf']   ).astype(np.int16)
    era['Td2']   = (era['Td2']   / era['T_sf']   ).astype(np.int16)
    era['T2']    = (era['T2']    / era['T_sf']   ).astype(np.int16)
    # humidity scaling
    era['q']     = (era['q']     / era['q_sf']   ).astype(np.uint16)
    # pressure scaling
    era['ps']    = (era['ps']    / era['ps_sf']  ).astype(np.uint16)
    # radiation scaling
    era['SW']    = (era['SW']    / era['rad_sf'] ).astype(np.uint16)
    era['LW']    = (era['LW']    / era['rad_sf'] ).astype(np.uint16)
    era['S_TOA'] = (era['S_TOA'] / era['rad_sf'] ).astype(np.uint16)
    # precipitation scaling
    era['P']     = (era['P']     / era['P_sf']   ).astype(np.uint16)
    # no scaling for geoportential height
    era['Z']     = era['Z'].astype(np.int16)

    return {'era': era}
