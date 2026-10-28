import os
import time
import warnings
from contextlib import nullcontext as _nullcontext
from itertools import product

import numpy as _np
import xarray as _xr
from dask.diagnostics import ProgressBar as _ProgressBar
from loguru import logger as _logger
from rasterio.warp import transform_bounds as _transform_bounds
from scipy.ndimage import gaussian_filter1d as _gaussian_filter1d

# importing this first raises a helpful error if the optional [data] dependencies are missing
from .from_planetary_computer import get_sentinel2_data

_Q = (0.05, 0.95)


def scl_to_snow(scl: _xr.DataArray) -> _xr.DataArray:
    """
    Convert Sentinel-2 scene classification (SCL) to a snow indicator.

    Parameters
    ----------
    scl : xarray.DataArray
        SCL data (can be lazy).

    Returns
    -------
    xarray.DataArray
        float32; 1 where snow/ice (11), 0 where vegetation, bare soil (4, 5, 6, 7)
        and NaN everywhere else (no data, saturated, shadows, cloud, cirrus, water).
    """
    snow = (scl == 11).astype("float32")
    return snow.where(scl.isin([4, 5, 6, 7, 11]))


def _bin_scatter(snow, bin_days):
    """Accumulate observations into (year, bin) sums using a one-hot matrix."""
    n_bins = 365 // bin_days
    years, yi = _np.unique(snow.time.dt.year.values, return_inverse=True)
    bi = _np.minimum((snow.time.dt.dayofyear.values - 1) // bin_days, n_bins - 1)

    onehot = _np.zeros((snow.sizes["time"], len(years) * n_bins), _np.float32)
    onehot[_np.arange(len(yi)), yi * n_bins + bi] = 1

    def scatter(a):
        out = a @ onehot  # (..., year * bin); unlike fancy assignment this accumulates
        return out.reshape(a.shape[:-1] + (len(years), n_bins))

    def to_year_bin(a):
        return _xr.apply_ufunc(
            scatter,
            a,
            input_core_dims=[["time"]],
            output_core_dims=[["year", "doy"]],
            dask="parallelized",
            dask_gufunc_kwargs={"output_sizes": {"year": len(years), "doy": n_bins}},
            output_dtypes=[_np.float32],
        ).assign_coords(year=years, doy=1 + _np.arange(n_bins) * bin_days)

    num = to_year_bin(snow.fillna(0))
    den = to_year_bin(snow.notnull().astype("float32"))
    return years, num, den


def _smooth_doy(a, sigma):
    return _xr.apply_ufunc(
        _gaussian_filter1d,
        a,
        input_core_dims=[["doy"]],
        output_core_dims=[["doy"]],
        kwargs={"sigma": sigma, "axis": -1, "mode": "wrap", "truncate": 4},
        dask="parallelized",
        output_dtypes=[_np.float32],
    )


def _boot_kernel(Sn, Sd, W, min_den, chunk, q):
    """Snow days for each bootstrap weight vector; returns (..., B) or quantiles (..., Q)."""
    W = W.reshape(W.shape[-2:])
    B = W.shape[0]
    out = _np.empty(Sn.shape[:-2] + (B,), _np.float32)
    for i in range(0, B, chunk):
        w = W[i : i + chunk]
        n, d = w @ Sn, w @ Sd
        with _np.errstate(invalid="ignore", divide="ignore"):
            p = _np.where(d >= min_den, n / d, _np.nan)
        out[..., i : i + chunk] = 365 * p.mean(-1)
    if q is None:
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels: all samples failed
        out = _np.nanquantile(out, q, axis=-1)
    return _np.moveaxis(out, 0, -1).astype(_np.float32)


def _snow_days_from_weights(Sn, Sd, W, min_den, chunk, q=None):
    n_out = W.sizes["boot"] if q is None else len(q)
    out_dim = "boot" if q is None else "quantile"
    out = _xr.apply_ufunc(
        _boot_kernel,
        Sn,
        Sd,
        W,
        input_core_dims=[["year", "doy"], ["year", "doy"], ["boot", "year"]],
        output_core_dims=[[out_dim]],
        kwargs={"min_den": min_den, "chunk": chunk, "q": q},
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {out_dim: n_out}},
        output_dtypes=[_np.float32],
    )
    return out if q is None else out.assign_coords(quantile=list(q))


def snow_days(
    snow: _xr.DataArray,
    bw=10,
    bin_days=5,
    min_den=1.0,
    n_boot=200,
    q=_Q,
    chunk=20,
    seed=0,
    compute=True,
    verbose=True,
) -> _xr.Dataset:
    """
    Number of snow days per year from sparse snow observations, with
    year-block bootstrap uncertainty.

    Observations of all years are binned by day of year, smoothed with a
    circular Gaussian kernel, and the snow probability per bin is averaged
    over bins (x 365). Bootstrap samples resample whole years with replacement.

    Parameters
    ----------
    snow : xarray.DataArray
        (time, y, x) with 1 = snow, 0 = no snow, NaN = no observation
        (see `scl_to_snow`).
    bw : float
        Gaussian smoothing bandwidth (sigma) in days.
    bin_days : int
        Width of the day-of-year bins in days. Day 366 is folded into the last bin.
    min_den : float
        Minimum smoothed observation count for a bin to be valid (else NaN).
    n_boot : int
        Number of bootstrap samples. Set to 0 to skip the uncertainty.
    q : tuple of float
        Quantiles of the bootstrap distribution to return.
    chunk : int
        Number of bootstrap samples processed at once (memory control).
    seed : int
        Random seed for the bootstrap.
    compute : bool
        Compute the result (otherwise returns a lazy dataset).
    verbose : bool
        Show a progress bar when computing.

    Returns
    -------
    xarray.Dataset
        ``n_snow_days`` (y, x), ``n_snow_days_q`` (quantile, y, x) and
        ``p_doy`` (doy, y, x), the snow probability per day-of-year bin.
    """
    snow = snow.transpose("time", ...)
    progress = _ProgressBar if verbose else _nullcontext
    if snow.chunks is not None:
        with progress():
            snow = snow.chunk(time=-1, y=64, x=64).persist()

    years, num, den = _bin_scatter(snow, bin_days)
    Y = len(years)
    sigma = bw / bin_days  # in bins
    Sn, Sd = _smooth_doy(num, sigma), _smooth_doy(den, sigma)
    min_den = min_den / (sigma * _np.sqrt(2 * _np.pi))  # gaussian_filter1d normalises the kernel

    def weights(w):
        return _xr.DataArray(w.astype(_np.float32), dims=("boot", "year"), coords={"year": years})

    point = _snow_days_from_weights(Sn, Sd, weights(_np.ones((1, Y))), min_den, chunk)
    ds = _xr.Dataset({"n_snow_days": point.isel(boot=0, drop=True)})

    n_all, d_all = Sn.sum("year"), Sd.sum("year")
    ds["p_doy"] = (n_all / d_all).where(d_all >= min_den * Y).transpose("doy", ...)

    if n_boot:
        rng = _np.random.default_rng(seed)
        W = rng.multinomial(Y, _np.full(Y, 1 / Y), size=n_boot)
        q = tuple(q)
        boots_q = _snow_days_from_weights(Sn, Sd, weights(W), min_den, chunk, q=q)
        # a few lucky resamples can pass where the point estimate fails; don't trust those
        ds["n_snow_days_q"] = boots_q.transpose("quantile", ...).where(ds.n_snow_days.notnull())

    if compute:
        with progress():
            ds = ds.compute()
    return ds


def _grid(bbox_WSEN, res, epsg):
    """Pixel (top-left) coordinates snapped outward to multiples of res, as stackstac does."""
    w, s, e, n = _transform_bounds("EPSG:4326", f"EPSG:{epsg}", *bbox_WSEN, densify_pts=21)
    x0, x1 = _np.floor(w / res) * res, _np.ceil(e / res) * res
    y0, y1 = _np.floor(s / res) * res, _np.ceil(n / res) * res
    x = x0 + res * _np.arange(round((x1 - x0) / res))
    y = y1 - res * _np.arange(round((y1 - y0) / res))
    return x, y


def _make_template(x, y, q, tile_px, epsg, attrs):
    import dask.array as da

    ny, nx = len(y), len(x)
    ntile = (-(-ny // tile_px), -(-nx // tile_px))
    nan = lambda *shape: da.full(shape, _np.nan, _np.float32, chunks=shape[:-2] + (tile_px,) * 2)
    ds = _xr.Dataset(
        {
            "n_snow_days": (("y", "x"), nan(ny, nx)),
            "n_snow_days_q": (("quantile", "y", "x"), nan(len(q), ny, nx)),
            "tile_done": (("tile_y", "tile_x"), _np.zeros(ntile, bool)),
        },
        coords={"y": y, "x": x, "quantile": list(q)},
        attrs=attrs,
    )
    return ds.rio.write_crs(f"EPSG:{epsg}")


def snow_days_tiled(
    dest,
    bbox_WSEN=(71, 36, 75, 40),
    res=100,
    years=range(2019, 2025),
    epsg=32643,
    tile_px=1024,
    max_cloud_cover=5,
    **kwargs,
) -> _xr.Dataset:
    """
    Compute snow days per year tile by tile and write them to a zarr store.

    The grid is split into non-overlapping blocks of ``tile_px`` x ``tile_px``
    pixels (in projected space). Finished tiles are flagged in ``tile_done``, so
    re-running with the same ``dest`` resumes where it stopped. Tiles for which
    no data can be fetched are skipped with a warning and retried on the next run.

    Parameters
    ----------
    dest : str or Path
        Local path of the zarr store. Reused if it already exists.
    bbox_WSEN : tuple
        Bounding box (West, South, East, North) in degrees.
    res : int
        Resolution in EPSG units.
    years : iterable of int
        Years of Sentinel-2 data to use.
    epsg : int
        EPSG code of the target grid.
    tile_px : int
        Tile size in pixels; also the zarr chunk size in y and x.
    max_cloud_cover : int
        Maximum scene cloud cover in percent.
    **kwargs
        Passed to `snow_days` (bw, bin_days, n_boot, q, ...).

    Returns
    -------
    xarray.Dataset
        The (lazily) opened zarr store with ``n_snow_days`` and ``n_snow_days_q``.
    """
    years = list(years)
    kwargs.setdefault("q", _Q)
    kwargs.setdefault("verbose", False)
    x, y = _grid(bbox_WSEN, res, epsg)

    if not os.path.exists(dest):
        attrs = {
            "years": years,
            "res": res,
            "bbox_WSEN": list(bbox_WSEN),
            "max_cloud_cover": max_cloud_cover,
        }
        attrs.update({k: list(v) if k == "q" else v for k, v in kwargs.items() if k != "verbose"})
        _make_template(x, y, kwargs["q"], tile_px, epsg, attrs).to_zarr(
            dest, compute=False, zarr_format=2
        )  # v3 does not reopen with this xarray
    ds = _xr.open_zarr(dest).set_coords("spatial_ref")
    done = ds["tile_done"].values.copy()

    tiles = list(product(range(done.shape[0]), range(done.shape[1])))
    t0 = time.time()
    for k, (j, i) in enumerate(tiles, 1):
        if done[j, i]:
            continue
        ys = slice(j * tile_px, min((j + 1) * tile_px, len(y)))
        xs = slice(i * tile_px, min((i + 1) * tile_px, len(x)))
        tx, ty = x[xs], y[ys]
        _logger.info(
            f"Tile {k}/{len(tiles)} (row {j}, col {i}); {(time.time() - t0) / 60:.1f} min elapsed"
        )

        # tile corners (top-left coords -> pixel edges) in lat/lon
        bbox = _transform_bounds(
            f"EPSG:{epsg}", "EPSG:4326", tx[0], ty[-1] - res, tx[-1] + res, ty[0], densify_pts=21
        )
        try:
            scl = get_sentinel2_data(
                tuple(bbox), years=years, res=res, epsg=epsg, max_cloud_cover=max_cloud_cover
            )
            scl = scl.reindex(x=tx, y=ty, method="nearest", tolerance=res / 2)
            out = snow_days(scl_to_snow(scl), **kwargs)
        except Exception as e:
            _logger.warning(f"Tile {k}/{len(tiles)} skipped: {e!r}")
            continue

        out = out[["n_snow_days", "n_snow_days_q"]].drop_vars(
            ["quantile", "spatial_ref"], errors="ignore"
        )
        out.to_zarr(dest, region={"y": ys, "x": xs})
        flag = _xr.Dataset({"tile_done": (("tile_y", "tile_x"), [[True]])})
        flag.to_zarr(dest, region={"tile_y": slice(j, j + 1), "tile_x": slice(i, i + 1)})
        done[j, i] = True

    return _xr.open_zarr(dest).set_coords("spatial_ref")
