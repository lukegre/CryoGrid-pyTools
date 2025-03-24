from typing import Union

import numpy as np
import xarray as xr


def read_OUT_regridded_file(fname: str, deepest_point=None) -> xr.Dataset:
    """
    Read a CryoGrid OUT_regridded[_FCI2] file and return it as an xarray dataset.

    Parameters
    ----------
    fname : str
        Path to the .mat file
    deepest_point : float, optional
        Represents the deepest depth of the profile relative to the surfface.
        If not provided, then elevation is returned. Negative values represent
        depths below the surface.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with dimensions 'time' and 'level'. The `elevation` coordinate
        represents the elevation above sea level based on the DEM. If
        deepest_point is provided, then an additional coordinate, `depth`,
        represents depth above surface.

    Notes
    -----
        For plotting, use `ds['variable'].plot(y='depth'/'elevation').
    """
    from cryogrid_pytools.matlab_helpers import (
        matlab2datetime,
        read_mat_struct_flat_as_dict,
    )

    dat = read_mat_struct_flat_as_dict(fname)

    for key in dat:
        dat[key] = dat[key].squeeze()

    ds = xr.Dataset()
    ds.attrs["filename"] = fname

    times = matlab2datetime(dat.pop("timestamp"))
    elev = dat.pop("depths")

    for key in dat:
        ds[key] = xr.DataArray(
            data=dat[key].astype("float32"),
            dims=["level", "time"],
            coords={"time": times},
        )

    ds = ds.chunk(dict(time=-1))

    ds["elevation"] = xr.DataArray(
        data=elev,
        dims=["level"],
        attrs={"units": "m", "long_name": "Elevation above sea level"},
    )

    ds = ds.set_coords("elevation")

    if deepest_point is not None:
        dz = deepest_point - ds["elevation"].min()
        ds["depth"] = (ds["elevation"] + dz).assign_attrs(
            units="m", long_name="Depth relative to surface"
        )
        ds = ds.set_coords("depth")

    ds = ds.chunk(dict(time=-1))

    return ds


def read_OUT_regridded_FCI2_file(fname: str, deepest_point=None) -> xr.Dataset:
    """
    Read a CryoGrid OUT_regridded_FCI2 file and return it as an xarray dataset.

    Parameters
    ----------
    fname : str
        Path to the .mat file
    deepest_point : float, optional
        Represents the deepest depth of the profile. If not provided,
        then elevation is returned. Negative values represent depths below
        the surface.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with dimensions 'time' and 'level'. The CryoGrid variable
        `depths` is renamed to `elevation`. If deepest_point is provided, then
        `depth` will represent the depth below the surface (negative below
        surface).
    """
    from warnings import warn

    warn(
        message="This function is deprecated. Use read_OUT_regridded_file instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )

    ds = read_OUT_regridded_file(fname, deepest_point)

    return ds


def _read_OUT_regridded_parallel(
    flist: list, deepest_point: float, **joblib_kwargs
) -> list:
    """
    Reads multiple files that are put out by the OUT_regridded class

    Parameters
    ----------
    flist: list
        List of file names that you want to read in.
    deepest_point: float
        When setting the configuration for when the data should be
        saved, the maximum depth is set. Give this number as a
        negative number here.
    concat_dim: str
        The dimension that the data should be concatenated along.
        Defaults to 'time', but 'gridcell' can also be used if
        the files are from different gridcells.
    joblib_kwargs: dict
        Uses the joblib library to do parallel reading of the files.
        Defaults are: n_jobs=-1, backend='threading', verbose=1

    Returns
    -------
    xr.Dataset
        An array with dimensions gridcell, depth, time.
        Variables depend on how the class was configured, but
        elevation will also be a variable.
    """

    import joblib

    # create the joblib tasks
    func = joblib.delayed(read_OUT_regridded_file)
    tasks = [func(f, deepest_point) for f in flist]

    # set up the joblib configuration
    joblib_props = dict(n_jobs=-1, backend="threading", verbose=1)
    joblib_props.update(joblib_kwargs)
    worker = joblib.Parallel(**joblib_props)  # type: ignore
    list_of_ds = list(worker(tasks))  # run the tasks

    return list_of_ds


def read_OUT_regridded_clusters(
    fname_glob: str,
    deepest_point: float,
    gridcell_func=lambda fname: fname.split("_")[-2],
    **joblib_kwargs,
) -> xr.Dataset:
    """
    Reads multiple files that are put out by the OUT_regridded class (and _FCI2)

    Parameters
    ----------
    fname_glob: str
        Path of the files that you want to read in.
        Use same notation as for glob(). Note that it expects
        name to follow the format `some_project_name_GRIDCELL_ID_date.mat`
        where GRIDCELL_ID will be extracted to assign the gridcell dimension.
        These GRIDCELL_IDs correspond with the index of the data in the
        flattened array.
    deepest_point: float
        When setting the configuration for when the data should be
        saved, the maximum depth is set. Give this number as a
        negative number here.
    joblib_kwargs: dict
        Uses the joblib library to do parallel reading of the files.
        Defaults are: n_jobs=-1, backend='threading', verbose=1

    Returns
    -------
    xr.Dataset
        An array with dimensions gridcell, depth, time.
        Variables depend on how the class was configured, but
        elevation will also be a variable.
    """
    import inspect

    from .utils import regex_glob

    # get the file list
    flist = regex_glob(fname_glob)
    # extract the gridcell from the file name
    gridcell = [gridcell_func(f) for f in flist]
    digits = [g.isdigit() for g in gridcell]

    if len(flist) == 0:
        raise FileNotFoundError(f"No files found with {fname_glob}")
    elif len(gridcell) != len(flist):
        raise ValueError(f"Could not extract gridcell from file names for {fname_glob}")
    elif not all(digits):
        not_digit = np.unique([f for f, d in zip(flist, digits) if not d])
        bad_func = "".join(inspect.getsource(gridcell_func).split("lambda")[1:]).strip()
        raise ValueError(
            f"Check your fname_glob ({fname_glob}) \n or gridcell_func ({bad_func}). \nGridcell ID "
            f"not a number for the following files:\n{not_digit}"
        )
    else:
        gridcell = [int(g) for g in gridcell]

    list_of_ds = _read_OUT_regridded_parallel(flist, deepest_point, **joblib_kwargs)

    # assign the gridcell dimension so that we can combine the data by coordinates and time
    list_of_ds = [ds.expand_dims(gridcell=[c]) for ds, c in zip(list_of_ds, gridcell)]
    ds = xr.combine_by_coords(list_of_ds, combine_attrs="drop_conflicts")

    assert isinstance(ds, xr.Dataset), "Something went wrong with the parallel reading."

    # transpose data so that plotting is quick and easy
    ds = ds.transpose("gridcell", "level", "time", ...)

    return ds


def make_fname(
    directory: str = r".",
    run_name: str = r"[0-9A-Za-z-_]{1,}",
    run_id: Union[str, int] = r"[0-9]{1,}",
    date: str = r"[12][0-9]{3}[0-2][0-9][0-3][0-9]",
    fname_format: str = "{run_name}_{run_id}_{date}.mat",
    **kwargs,
) -> str:
    """
    Create a file name from a format string and a dictionary of keyword arguments.

    Parameters
    ----------
    directory : str, optional
        The directory where the file is located.
    run_name : str, optional
        The name of the run.
    run_id : Union[str, int], optional
        The cluster centroid index number from MATLAB.
    date : str, optional
        The date of the run in the format 'YYYYMMDD' (valid for 1000-01-01 to 2999-12-31).
    fname_format : str, optional
        The format string for the file name. It should contain keys in curly braces.
        Defaults to the OUT_regridded standard format
    **kwargs : dict
        Additional keyword arguments to be used in the format string.

    Returns
    -------
    str
        The formatted string.
    """
    import pathlib
    import re

    # get format key sin fname_format
    regex = re.compile(r"{(.*?)}")
    keys = regex.findall(fname_format)

    default_kwargs = dict(
        run_name=run_name,
        run_id=run_id,
        date=date,
    )
    kwargs = default_kwargs | kwargs

    # check if all keys are in kwargs
    missing_keys = [k for k in keys if k not in kwargs]
    if len(missing_keys) > 0:
        raise ValueError(f"Missing keys in kwargs: {missing_keys}")

    # format the string
    fname = fname_format.format(**kwargs)
    fname = str(pathlib.Path(directory).expanduser().resolve() / fname)

    return fname
