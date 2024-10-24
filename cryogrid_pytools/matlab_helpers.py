# standalone file that can be shared without the rest of the package
from loguru import logger  
import pandas as pd
import xarray as xr
import numpy as np


def read_mat_struct_as_dataset(fname, index=None):
    from .matlab_helpers import read_mat_struct_flat_as_dict
    
    out = read_mat_struct_flat_as_dict(fname)
    df = pd.DataFrame.from_dict(out)
    if index is not None:
        df = df.set_index(index)
    ds = df.to_xarray()

    return ds


def read_mat_struct_flat_as_dict(fname: str) -> dict:
    """
    Read a MATLAB struct from a .mat file and return it as a dictionary.
    
    Assumes that the struct is flat, i.e. it does not contain any nested
    structs.

    Parameters
    ----------
    fname : str
        Path to the .mat file
    
    Returns
    -------
    data : dict
        Dictionary with the struct fields as keys and the corresponding
        data as values.
    """
    from scipy.io import loadmat

    raw = loadmat(fname)
    key = [k for k in raw.keys() if not k.startswith('_')][0]
    array_with_names = raw[key][0]

    names = array_with_names.dtype.names
    arrays = [a.squeeze() for a in array_with_names[0]]
    data = {k: v for k, v in zip(names, arrays)}

    return data


def datetime2matlab(time: xr.DataArray, reference_datestr:str="1970-01-01") -> np.ndarray:
    """
    Converts the time dimension of a xarray dataset to matlab datenum format

    Parameters
    ----------
    time_hrs : xr.DataArray
        Time from dataset, but only supports hour resolution and lower (days, months, etc)
    reference_datestr : str
        Reference date string in format 'YYYY-MM-DD'. In many cases this is 1970-01-01

    Returns
    -------
    np.ndarray
        Array of matlab datenum values
    """

    def get_matlab_datenum_offset(reference_datestr):
        """
        Returns the matlab datenum offset for a given reference date string
        """
        
        # this is hard coded in matlab, which uses 0000-01-01 as the reference date
        # but this isn't a valid date in pandas, so we use -0001-12-31 instead
        matlab_t0 = pd.Timestamp('-0001-12-31')  
        reference_date = pd.Timestamp(reference_datestr)
        offset_days = (matlab_t0 - reference_date).days
        
        return offset_days
        
    hours_since_ref = time.values.astype('datetime64[h]').astype(float)
    days_since_ref = hours_since_ref / 24

    matlab_offset = get_matlab_datenum_offset(reference_datestr)
    matlab_datenum = days_since_ref - matlab_offset

    return matlab_datenum


def matlab2datetime(matlab_datenum):
    """
    Convert a MATLAB datenum to a pandas.Timestamp

    Parameters
    ----------
    matlab_datenum : float
        MATLAB datenum
    
    Returns
    -------
    pd.Timestamp
        Timestamp object
    """

    matlab_epoch = 719529
    timestamps = pd.to_datetime(matlab_datenum - matlab_epoch, unit='D')

    return timestamps