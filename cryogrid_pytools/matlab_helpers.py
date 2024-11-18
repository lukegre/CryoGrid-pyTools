# standalone file that can be shared without the rest of the package
from loguru import logger  
import pandas as pd
import xarray as xr
import numpy as np


def read_mat_struct_as_dataset(fname, index=None, index_is_datenum=False):
    """
    Read a MATLAB struct from a .mat file and return it as an xarray dataset.

    Parameters
    ----------
    fname : str
        Path to the .mat file. All variables in the struct are assumed to have 
        the same dimensions and shape (except for the index columns).
    index : str, tuple, optional [default=None]
        Name of the index column. If None is passed [default], then no index is set.
        If a tuple is passed, then the corresponding columns are used as a multiindex.
    index_is_datenum : bool, optional [default=False]
        If True, then the index is converted from MATLAB datenum to a pandas.Timestamp.
    
    Returns
    -------
    ds : xr.Dataset
        Dataset with the struct fields as variables and the corresponding
        data as values. 
    """
    out = read_mat_struct_flat_as_dict(fname)
    df = pd.DataFrame.from_dict(out)
    if index is not None:
        df = df.set_index(index)
    ds = df.to_xarray()

    if index_is_datenum:
        assert index is not None, "index must be set if index_is_matlab_datenum is True"
        assert isinstance(index, str), "index must be a single index (dtype string) if index_is_matlab_datenum is True"
        ds = ds.assign_coords(**{index: matlab2datetime(ds[index].values)})

    return ds


def read_mat_struct_flat_as_dict(fname: str, key=None) -> dict:
    """
    Read a MATLAB struct from a .mat file and return it as a dictionary.
    
    Assumes that the struct is flat, i.e. it does not contain any nested
    structs.

    Parameters
    ----------
    fname : str
        Path to the .mat file
    key : str, optional
        The name of the matlab key in the .mat file. If None is passed [default],
        then the first key that does not start with an underscore is used.
        If a string is passed, then the corresponding key is used.
    
    Returns
    -------
    data : dict
        Dictionary with the struct fields as keys and the corresponding
        data as values.
    """
    from scipy.io import loadmat

    raw = loadmat(fname)
    
    keys = [k for k in raw.keys() if not k.startswith('_')]

    if key is None:
        logger.log(5, f"No key specified. Using first key that does not start with an underscore: {keys[0]}")
        key = keys[0]
    elif key not in keys:
        raise ValueError(f"Key '{key}' not found in .mat file. Available keys are: {keys}")
    
    named_array = unnest_matlab_struct_named_array(raw[key])
    data = {k: named_array[k].squeeze() for k in named_array.dtype.names}

    return data
    


def unnest_matlab_struct_named_array(arr: np.ndarray) -> dict:
    """
    Unnest a numpy structured array that was read from a MATLAB .mat file.

    Parameters
    ----------
    arr : np.ndarray
        Structured array read from a .mat file
    
    Returns
    -------
    arr : np.ndarray
        Unnested named array where arr.dtype.names has a length > 1 
        and arr[i] corresponds with arr.dtype.names[i]. Note that 
    
    Note
    ----
    This function works with two of my examples, but may be a bit buggy. 
    It is not well tested and may not work in all cases.
    """
    def is_ndarray_or_void(x):
        return isinstance(x, np.ndarray) or isinstance(x, np.void)
    
    while is_ndarray_or_void(arr) and arr.size == 1:
        if (  # stop if a void array with multiple fields
            isinstance(arr, np.void) 
            and (arr.dtype.names is not None) 
            and (len(arr.dtype.names) > 1)):
            return arr
        # otherwise continue
        prev = arr
        arr = prev[0]

    return prev


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