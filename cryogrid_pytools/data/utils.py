from memoization import cached as _cached
import xarray as _xr

import earthaccess as _earthaccess
import pooch as _pooch
import rioxarray as _rxr  # noqa
import xarray_raster_vector as _xrv  # noqa


def _decorator_dataarray_to_bbox(func):
    """
    A decorator that processes the first argument of the decorated function to handle
    either an xarray DataArray or a bounding box tuple. If the first argument is a
    DataArray, it extracts the bounding box from the DataArray and reprojects the
    output to match the DataArray's projection.

    Parameters
    ----------
    func : callable
        The function to be decorated. The function should accept a
        bounding box as its first argument.

    Returns
    -------
    callable
        The wrapped function with the additional functionality of handling
        DataArray and reprojecting the output.
    """

    @_cached
    def wrapper(*args, **kwargs):
        if len(args) >= 1:
            bbox_or_target = args[0]
            args = args[1:]
        else:
            bbox_or_target = kwargs.pop("bbox", None) or kwargs.pop("bbox_WSEN", None)

        if isinstance(bbox_or_target, _xr.DataArray):
            da = bbox_or_target
            bbox = da.rv.get_bbox_latlon()
            out = func(bbox, *args, **kwargs)
            out = out.rio.reproject_match(da)

        elif isinstance(bbox_or_target, tuple):
            bbox = bbox_or_target
            out = func(bbox_or_target, *args, **kwargs)

        else:
            message = (
                f"The first argument must be a bounding box tuple or an xarray DataArray, "
                f"but got {type(bbox_or_target)} instead."
            )

            raise ValueError(message)

        return out

    return wrapper


def get_earthaccess_session():
    """
    Logs into earthaccess and gets session info.

    Returns
    -------
    session
        The session information.
    """

    auth = _earthaccess.login(persist=True)
    session = auth.get_session()

    return session


def download_url(url, **kwargs):
    """
    Download a file from a given URL and process it if necessary.

    Parameters
    ----------
    url : str
        The URL of the file to download.
    **kwargs : Additional properties passed to pooch.retrieve.
        These properties will override the default settings.

    Returns
    -------
    list
        A list of file paths to the downloaded (and possibly
        decompressed) files.
    """

    if url.endswith(".zip"):
        processor = _pooch.Unzip()
    elif url.endswith(".tar.gz"):
        processor = _pooch.Untar()
    else:
        processor = None

    default_props = dict(
        known_hash=None,
        fname=url.split("/")[-1],
        path=None,
        downloader=_pooch.HTTPDownloader(progressbar=True),
        processor=processor,
    )

    props = default_props | kwargs
    flist = _pooch.retrieve(url=url, **props)

    return flist


def check_epsg(epsg: int) -> bool:
    """
    Check if the provided EPSG code is valid.

    Parameters
    ----------
    epsg : int
        The EPSG code to be checked.

    Returns
    -------
    bool
        True if the EPSG code is valid, False otherwise.

    Raises
    ------
    AssertionError
        If the EPSG code is not valid.
    """

    def check_epsg(epsg: int) -> bool:
        """
        Check if the provided EPSG code is valid.

        This function checks whether the given EPSG code is either a UTM code
        (starting with '326') or the Lat/Lon code (4326).

        Args:
            epsg (int): The EPSG code to be checked.

        Returns:
            bool: True if the EPSG code is valid, False otherwise.

        Raises:
            AssertionError: If the EPSG code is not valid.
        """

    is_valid_epsg = str(epsg).startswith("326") or (epsg == 4326)
    assert is_valid_epsg, "The EPSG code must be UTM (326xx) or Lat/Lon (4326)."


def get_res_in_proj_units(res_m, epsg, min_res=30):
    """
    Check if the resolution is valid for the given EPSG code.

    Parameters
    ----------
    res_m : int
        The resolution in meters.
    epsg : int
        The EPSG code of the projection.
    min_res : int
        The minimum resolution required for the dataset.

    Returns
    -------
    res : int
        The resolution in the units of the projection.
    """
    message = f"The resolution must be greater than {min_res}m for this collection"
    assert res_m > min_res, message
    res = res_m / 111111 if epsg == 4326 else res_m

    return res


def smooth_data(da: _xr.DataArray, kernel_size: int = 3, n_iters=1) -> _xr.DataArray:
    """
    Smooth the data using a rolling mean filter (box kernel).

    Parameters
    ----------
    da : xarray.DataArray
        The input data as an xarray DataArray.
    kernel_size : int
        The size of the kernel for the rolling mean filter.
    n_iters : int
        The number of iterations to apply the filter.

    Returns
    -------
    xarray.DataArray
        The smoothed data as an xarray DataArray.
    """

    da_smooth = da.copy()
    for _ in range(n_iters):
        da_smooth = da_smooth.rolling(
            x=kernel_size, y=kernel_size, center=True, min_periods=1
        ).mean()

    if n_iters:
        da_smooth = da_smooth.assign_attrs(
            smoothing_kernel="box_kernel",
            smoothing_kernel_size=kernel_size,
            smoothing_iterations=n_iters,
        )

    return da_smooth


def read_zenodo_record(zenodo_id, flist=None, dest_dir=None):
    """
    Downloads files from a Zenodo record and optionally filters them by a specified list of filenames.

    Parameters
    ----------
    zenodo_id : str
        The Zenodo record ID to retrieve files from.
    flist : list of str, optional
        A list of filenames to filter the files to be downloaded. If None, all files in the record will be downloaded.
    dest_dir : str, optional
        The directory where the downloaded files will be stored. If None, the files will be stored in the default cache directory.

    Returns
    -------
    list of str
        A list of file paths to the downloaded (and possibly extracted) files.

    Raises
    ------
    Exception
        If there are issues accessing the Zenodo record or downloading the files.

    Examples
    --------
    >>> read_zenodo_record("10732042", ["file1.shp", "file2.shp"])
    ['/path/to/file1.shp', '/path/to/file2.shp']
    """
    import pyzenodo3

    # Access the Zenodo record
    z = pyzenodo3.Zenodo()
    record = z.get_record(zenodo_id)

    files = record.data["files"]

    out = []
    for i, f in enumerate(files):
        name = f["key"]
        if flist is not None:
            if name not in flist:
                continue
        link = f["links"]["self"]

        # Handle ZIP files and other formats
        if name.endswith(".zip"):
            processor = _pooch.Unzip()
        else:
            processor = None

        fname = _pooch.retrieve(
            link, None, fname=name, path=dest_dir, processor=processor
        )
        if isinstance(fname, list):
            out += fname
        else:
            out.append(fname)
    return out
