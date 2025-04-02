import numpy as np
import pandas as pd
import xarray as xr
from typing import Union


def read_spatial_data(
    fname_spatial_mat: str, crs: Union[str, None] = None, gridcell_name="matlab_index"
) -> xr.Dataset:
    """
    Opens run_spatial_info.mat that contains the spatial information for the run.

    Note
    ----
    You have to manually save this data since Python cannot read in
    the run_parameters.mat file. It needs to be a flat struct. Below is the
    MATLAB code to save the spatial data:

    ```matlab
    cluster_num = run_info.CLUSTER.STATVAR.cluster_number;

    data.coord_x = run_info.SPATIAL.STATVAR.X;
    data.coord_y = run_info.SPATIAL.STATVAR.Y;
    data.lat = run_info.SPATIAL.STATVAR.latitude;
    data.lon = run_info.SPATIAL.STATVAR.longitude;

    data.mask = run_info.SPATIAL.STATVAR.mask;

    data.elevation = run_info.SPATIAL.STATVAR.altitude;
    data.slope_angle = run_info.SPATIAL.STATVAR.slope_angle;
    data.aspect = run_info.SPATIAL.STATVAR.aspect;
    data.skyview_factor = run_info.SPATIAL.STATVAR.skyview_factor;
    data.stratigraphy_index = run_info.SPATIAL.STATVAR.stratigraphy_index;
    data.matlab_index = [1 : size(data.elevation, 1)]';

    data.cluster_num = run_info.CLUSTER.STATVAR.cluster_number;
    data.cluster_idx = run_info.CLUSTER.STATVAR.sample_centroid_index;

    sname = strcat(provider.PARA.result_path, provider.PARA.run_name, '/run_spatial_info.mat');
    save(sname, 'data');
    ```

    Parameters
    ----------
    fname_spatial_mat : str
        Path to the run_spatial_info.mat file
    crs : str, optional
        Coordinate reference system of the spatial data (e.g., EPSG:32633)
        Get this from the DEM used in your run. If None, no CRS is added,
        by default None
    gridcell_name : str, optional
        Name of the gridcell variable in the spatial data, by default 'matlab_index'
        This is based on how you save the spatial data in MATLAB

    Returns
    -------
    ds : xr.Dataset
        Dataset with the spatial information. Contains the variables:
        1D variables [cluster_num as dim]:
            - cluster_centroid_gridcell:
                    the gridcell of the cluster centroid of
                    each cluster with the cluster_num as the dimension
        2D variables [y, x as dims]:
            - lon / lat: longitude and latitude of the gridcells
            - mask: mask of the gridcells
            - elevation-based variables (e.g., elevation, slope, aspect)
            - stratigraphy_index: used for stratigraphy configuration selection
            - gridcell: the index of flattened data represented in 2D
            - cluster_num_2d: the cluster number that each gridcell belongs to
            - cluster_centroid_gridcell_2d: the gridcell of the cluster centroid of each cluster

    """
    import cryogrid_pytools as cg

    spatial_dict = cg.read_mat_struct_flat_as_dict(fname_spatial_mat)

    # remove the cluster_idx since it has different dimensions
    centroid_gridcell = spatial_dict.pop("cluster_idx")

    # convert to DataFrame and then to xarray with [y, x] as dimensions
    ds = (
        pd.DataFrame.from_dict(spatial_dict)
        .set_index(["coord_y", "coord_x"])
        .to_xarray()
    )

    # some house-keeping for the dataset casting things to the right dtypes
    ds["mask"] = ds["mask"].fillna(0).astype("bool")
    for var in ["cluster_num", "stratigraphy_index", gridcell_name]:
        ds[var] = ds[var].astype("uint32")
    ds = ds.rename({gridcell_name: "gridcell", "cluster_num": "cluster_num_2d"})

    # add the centroid_idx as a coordinate with cluster_num as index (0 + 1 - k)
    ds["cluster_centroid_gridcell"] = xr.DataArray(
        data=np.r_[0, centroid_gridcell].astype(
            "uint32"
        ),  # prepending 0 for masked data
        coords={"cluster_num": np.arange(centroid_gridcell.size + 1)},  # k + 1
        dims=("cluster_num",),  # cluster_num is the equivalent to cluster_num
        attrs=dict(
            description=(
                "gridcell of the cluster centroid of each cluster, where "
                "the gridcell represents a flattened index. 0 is used for "
                "masked data."
            ),
            long_name="Cluster centroid gridcell",
        ),
    )

    # cluster_centroid_gridcell [0-k] has cluster number index, we can thus
    # use the cluster_num to convert this flat data to 2D gridcell data
    ds["cluster_centroid_gridcell_2d"] = (
        ds["cluster_centroid_gridcell"]
        .sel(cluster_num=ds.cluster_num_2d)
        .astype("uint32")
        .assign_attrs(
            description=(
                "Each pixel belongs to a cluster. Each cluster has a centroid "
                "that represents that entire cluster. This array gives the "
                "gridcell of the centroid mapped out to the cluster."
            )
        )
    )

    ds = ds.rename(coord_x="x", coord_y="y")
    if crs is not None:
        ds = ds.rio.write_crs(crs)

    return ds


def map_gridcells_to_clusters(
    da: xr.DataArray, cluster_centroid_gridcell_2D: xr.DataArray
) -> xr.DataArray:
    """
    Maps the single depth selection of the profiles to the 2D clusters

    Parameters
    ----------
    da : xr.DataArray
        Single depth selection of the profiles with gridcell dimension only.
        Note that gridcell must start at 1 (0 is reserved for masked data).
    cluster_centroid_gridcell_2D : xr.DataArray
        2D array with the gridcell of the cluster centroid of each cluster
        Must have dtype uint32. Can have 0 to represent masked data.

    Returns
    -------
    da_2d_mapped : xr.DataArray
        The 2D array of the profiles mapped to the clusters with the same shape as
        cluster_centroid_gridcell_2D. The 2D gridcells will also be given as a coordinate

    Raises
    ------
    ValueError
        If da (variable to be mapped) does not have gridcell dimension only
    """

    # make sure profiles_single_depth has gridcell dimension only
    if list(da.sizes) != ["gridcell"]:
        raise ValueError(
            "da must have gridcell dimension only (i.e., single timestep and depth)"
        )

    # if there is masked data in the cluster_centroid_gridcell_2D, then we need to
    # create a dummy gridcell, otherwise an error will be raised when using the
    # .sel() method below
    if 0 in cluster_centroid_gridcell_2D:
        # create a single gridcell with 0 value
        dummy0 = xr.DataArray(data=0, dims=("gridcell",), coords=dict(gridcell=[0]))
        # concatenate the dummy gridcell to the cluster_centroid_gridcell_2D
        da = xr.concat([dummy0, da], dim="gridcell").assign_attrs(**da.attrs)

    # use the cluster_centroid_gridcell_2D to map the gridcell to the spatial clusters
    da_2d_mapped = da.sel(gridcell=cluster_centroid_gridcell_2D).where(lambda x: x != 0)

    da_2d_mapped.attrs = da.attrs
    da_2d_mapped.attrs["history"] = (
        da.attrs.get("history", "")
        + f"Mapped to 2D clusters using {cluster_centroid_gridcell_2D.name}"
    )

    return da_2d_mapped
