from typing import Union

import numpy as np
import pandas as pd
import xarray as xr


def read_spatial_data(
    fname_spatial_mat: str,
    crs: Union[str, None] = None,
) -> xr.Dataset:
    """
    Opens run_spatial_info.mat that contains the spatial information for the run.

    Note
    ----
    You have to manually save this data since Python cannot read in
    the run_parameters.mat file. It needs to be a flat struct. Below is the
    MATLAB code to save the spatial data:

    ```matlab
    skip_fields = {'key'};
    for field = fieldnames(run_info.SPATIAL.STATVAR)'
        field_item_spatial = field{1}; % Extract string
        if ~ismember(field_item_spatial, skip_fields)
            data.(field_item_spatial) = run_info.SPATIAL.STATVAR.(field_item_spatial);
        end
    end

    for field = fieldnames(run_info.CLUSTER.STATVAR)'
        field_item_cluster = field{1}; % Same fix here
        data.(field_item_cluster) = run_info.CLUSTER.STATVAR.(field_item_cluster);
    end

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

    Returns
    -------
    ds : xr.Dataset
        Dataset with the spatial information. Contains the variables:
        1D variables [cluster_num as dim]:
            - cluster_centroid_index:
                    the index of the cluster centroid of
                    each cluster with the cluster_num as the dimension
        2D variables [y, x as dims]:
            - longitude / latitude
            - mask: mask of the indexs
            - all variables in run_info.SPATIAL.STATVAR
            - from run_info.CLUSTER.STATVAR
                - cluster_number: the cluster number that each index belongs to
                - cluster_number_mapped: a 2d representation of the cluster_number
                - cluster_centroid_index: the index of the cluster centroid with cluster_number as the dim
                - cluster_centroid_index_mapped: the index of the cluster centroid of each cluster

    """
    import cryogrid_pytools as cg

    spatial_dict = cg.read_mat_struct_flat_as_dict(fname_spatial_mat)

    # in the last version, the original matlab names weren't used.
    # now, I've gone back to using MATLAB CryoGrid naming conventions
    # so, for backwards compatibility, we need to rename the variables
    # from the previous approach to match the MATLAB names again.
    # matlab_index is a variable that only exists in the old version
    if "matlab_index" in spatial_dict:
        spatial_dict = renamer_old_to_new(spatial_dict)

    # remove the sample_centroid_index since it has different dimensions
    centroid_index = spatial_dict.pop("sample_centroid_index")

    spatial_dict["matlab_index"] = np.arange(1, spatial_dict["mask"].size + 1)
    # convert to DataFrame and then to xarray with [y, x] as dimensions
    ds = (
        pd.DataFrame.from_dict(spatial_dict)
        .set_index(["Y", "X"])
        .to_xarray()
        .rename(cluster_number="cluster_number_mapped")
    )

    # add the centroid_idx as a coordinate with cluster_number as index (0 + 1 - k)
    ds["cluster_centroid_index"] = xr.DataArray(
        data=np.r_[0, centroid_index].astype("uint32"),  # prepending 0 for masked data
        coords={"cluster_number": np.arange(centroid_index.size + 1)},  # k + 1
        dims=("cluster_number",),
        attrs=dict(
            description=(
                "index of the cluster centroid of each cluster, where "
                "the index represents a flattened index. 0 is used for "
                "masked data."
            ),
            long_name="Cluster centroid index",
        ),
    )

    # cluster_centroid_index_mapped [0-k] has cluster number index, we can thus
    # use the cluster_num to convert this flat data to 2D index data
    ds["cluster_centroid_index_mapped"] = (
        ds["cluster_centroid_index"]
        .sel(cluster_number=ds.cluster_number_mapped.fillna(0))
        .astype("uint32")
        .assign_attrs(
            description=(
                "Each pixel belongs to a cluster. Each cluster has a centroid "
                "that represents that entire cluster. This array gives the "
                "index of the centroid mapped out to the cluster."
            )
        )
    )

    ds = ds.rename(X="x", Y="y")
    if crs is not None:
        ds = ds.rio.write_crs(crs)

    return ds


def renamer_old_to_new(spatial_dict):
    """
    Renames the variables in the spatial_dict to match the new naming
    conventions.

    In the last version, the original matlab names weren't used.
    Now, I've gone back to using MATLAB CryoGrid naming conventions so,
    for backwards compatibility, we need to rename the variables from
    the previous approach to match the MATLAB names again. matlab_index
    is a variable that only exists in the old version.

    Parameters
    ----------
    spatial_dict : dict
        Dictionary with the spatial data. The keys are the variable names
        and the values are the data.

    Returns
    -------
    spatial_dict : dict
        Dictionary with the renamed variables. The keys are the new variable names
        and the values are the data.
    """
    renamer = dict(
        coord_x="X",
        coord_y="Y",
        lat="latitude",
        lon="longitude",
        cluster_num="cluster_number",
        cluster_idx="sample_centroid_index",
    )
    for old_name in renamer:
        new_name = renamer[old_name]
        spatial_dict[new_name] = spatial_dict.pop(old_name)

    return spatial_dict


def map_gridcells_to_clusters(
    da: xr.DataArray, cluster_centroid_index_mapped: xr.DataArray, missing_value=np.nan
) -> xr.DataArray:
    """
    Maps the single depth selection of the profiles to the 2D clusters

    Parameters
    ----------
    da : xr.DataArray
        Single depth selection of the profiles with `index` dimension only.
        Note that `index` must start at 1 (0 is reserved for masked data).
    cluster_centroid_index_mapped : xr.DataArray
        2D array with the index of the cluster centroid of each cluster
        Must have dtype uint32. Can have 0 to represent masked data.

    Returns
    -------
    da_2d_mapped : xr.DataArray
        The 2D array of the profiles mapped to the clusters with the same shape as
        cluster_centroid_index_mapped. The 2D gridcells will also be given as a coordinate

    Raises
    ------
    ValueError
        If da (variable to be mapped) does not have index dimension only
    """

    # make sure profiles_single_depth has index dimension only
    if list(da.sizes) != ["index"]:
        raise ValueError(
            "da must have `index` dimension only (i.e., single timestep and depth)"
        )
    if da.index.isin([0]).any():
        raise ValueError("`da.index` must start at 1 (0 is reserved for masked data).")

    missing_clusters = ~cluster_centroid_index_mapped.isin(
        [0] + da.index.values.tolist()
    )
    cluster_centroid_index_mapped = (
        cluster_centroid_index_mapped.where(~missing_clusters)
        .fillna(0)
        .astype("uint32")
    )

    # if there is masked data in the cluster_centroid_index_mapped, then we need to
    # create a dummy index, otherwise an error will be raised when using the
    # .sel() method below
    if 0 in cluster_centroid_index_mapped:
        # create a single index with 0 value
        dummy0 = xr.DataArray(data=0, dims=("index",), coords=dict(index=[0]))
        # concatenate the dummy index to the cluster_centroid_index_mapped
        da = xr.concat([dummy0, da], dim="index").assign_attrs(**da.attrs)

    # use the cluster_centroid_index_mapped to map the index to the spatial clusters
    da_2d_mapped = da.sel(index=cluster_centroid_index_mapped)
    da_2d_mapped = da_2d_mapped.where(lambda x: x != 0).where(
        ~missing_clusters, missing_value
    )

    da_2d_mapped.attrs = da.attrs
    da_2d_mapped.attrs["history"] = (
        da.attrs.get("history", "")
        + f"Mapped to 2D clusters using {cluster_centroid_index_mapped.name}"
    )

    return da_2d_mapped
