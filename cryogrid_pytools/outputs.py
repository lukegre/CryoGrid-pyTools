# depends on matlab_helpers.py
import xarray as xr
import pandas as pd
import numpy as np


def read_OUT_regridded_FCI2_cluster_file(fname):
    from .matlab_helpers import read_mat_struct_flat_as_dict, matlab2datetime

    gridcell = int(fname.split('_')[-2])

    data = read_mat_struct_flat_as_dict(fname)
    
    elevation = data.pop('depths').squeeze()
    time = data.pop('timestamp').squeeze()
    coords = {'time': time}

    ds = xr.Dataset()
    for key in data:
        arr = data[key]
        ds[key] = xr.DataArray(
            data=arr,
            dims=('depth', 'time'),
            coords=coords)
    
    ds = ds.chunk({})
    ds = ds.expand_dims(gridcell=[gridcell])
    ds = ds.isel(time=slice(0, -1))
    
    ds['elevation'] = xr.DataArray([elevation], dims=('gridcell','depth'))
    ds = ds.assign_coords(time=ds.time.to_series().apply(matlab2datetime))
    ds = ds.transpose('gridcell', 'time', 'depth', ...).astype('float32')
        
    return ds


def read_OUT_regridded_FCI2_cluster_parallel(glob_fname, depth_minmax=[1.5, -5], **joblib_kwargs):
    from glob import glob
    import joblib

    flist = sorted(glob(glob_fname))
    
    joblib_props = dict(n_jobs=-1, backend='threading', verbose=1)
    joblib_props.update(joblib_kwargs)
    
    func = joblib.delayed(read_OUT_regridded_FCI2_cluster_file)
    tasks = [func(f) for f in flist]
    worker = joblib.Parallel(**joblib_props)
    
    output = worker(tasks)
    ds = xr.combine_by_coords(output)

    ds = ds.assign_coords(
        depth=np.linspace(depth_minmax[0], depth_minmax[1], ds.depth.size))
    
    ds = ds.transpose('gridcell', 'time', 'depth', ...)
    ds = ds.assign(elevation=ds.elevation.mean('time'))

    return ds


def read_OUT_regridded_FCI2_point_file(fname, max_depth=5):
    from .matlab_helpers import read_mat_struct_flat_as_dict, matlab2datetime
    
    dat = read_mat_struct_flat_as_dict(fname)
    for key in dat:
        dat[key] = dat[key].squeeze()

    elev = dat.pop('depths')
    depth = elev - elev.min() - max_depth
    elev = elev[np.argmin(np.abs(depth))]
    times = matlab2datetime(dat.pop('timestamp'))

    ds = xr.Dataset()
    for key in dat:
        ds[key] = xr.DataArray(dat[key], dims=['depth', 'time'], coords={'time': times, 'depth': depth})

    ds.attrs['elevation'] = elev
    ds.attrs['filename'] = fname
    ds.transpose('time', 'depth', ...)

    return ds