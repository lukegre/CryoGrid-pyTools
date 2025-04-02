# API Reference

This page contains the detailed API reference for CryoGrid-pyTools.

## Excel Config file

::: cryogrid_pytools.CryoGridConfigExcel
    handler: python
    options:
      members:
        - get_class
        - get_classes
        - get_coord_path
        - get_dem_path
        - get_output_max_depth

## Reading profile outputs

::: cryogrid_pytools.read_OUT_regridded_file
::: cryogrid_pytools.read_OUT_regridded_files
::: cryogrid_pytools.read_mat_struct_flat_as_dict
::: cryogrid_pytools.read_mat_struct_as_dataset

## Reading clustering outputs
::: cryogrid_pytools.spatial_clusters.read_spatial_data
::: cryogrid_pytools.spatial_clusters.map_gridcells_to_clusters

## ERA5 Forcing

::: cryogrid_pytools.forcing.read_mat_ear5
::: cryogrid_pytools.forcing.era5_to_matlab

## Elevation, land cover, snow melt

::: cryogrid_pytools.data.get_dem_copernicus30
::: cryogrid_pytools.data.get_esa_land_cover
::: cryogrid_pytools.data.get_snow_melt_doy
::: cryogrid_pytools.data.get_randolph_glacier_inventory
