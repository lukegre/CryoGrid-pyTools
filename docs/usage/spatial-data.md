# Spatial Data and Forcing

The `data` module provides tools for creating forcing data and spatial data for CryoGrid spatial cluster runs. This module requires additional dependencies which can be installed using:

```bash
pip install "cryogrid_pytools[data]"
```

## Digital Elevation Model (DEM)

You can download DEM data from the Copernicus 30m dataset:

```python
import cryogrid_pytools as cgt

# Define your area of interest (West, South, East, North)
bbox = [70.0, 35.0, 71.0, 36.0]  # Example for a region in the Pamirs

# Get DEM data at 30m resolution
dem = cgt.data.get_dem_copernicus30(
    bbox_WSEN=bbox,
    res_m=30,
    epsg=32643,  # UTM 43N (default for Pamir region)
    smoothing_iters=2,  # Apply smoothing to reduce noise
    smoothing_size=3    # Kernel size for smoothing
)
```

## Land Cover Data

Get ESA World Cover data for your region:

```python
landcover = cgt.data.get_esa_land_cover(
    bbox_WSEN=bbox,
    res_m=30,
    epsg=32643
)
```

The returned DataArray includes attributes for class values, descriptions, and colors that can be used for plotting.

## Snow Melt Timing

Calculate snow melt timing using Sentinel-2 data:

```python
# Get snow melt day of year for multiple years
snow_melt = cgt.data.get_snow_melt_doy(
    bbox_WSEN=bbox,
    years=range(2018, 2025),  # Analysis period
    res_m=30,
    epsg=32643
)
```

## Glacier Data

Get Randolph Glacier Inventory (RGI) data for your region:

```python
# Get glacier data as a raster matching your DEM
glacier_data = cgt.data.get_randolph_glacier_inventory(target_dem=dem)

# Or get raw vector data
glacier_vector = cgt.data.get_randolph_glacier_inventory()
```

## ERA5 Forcing Data

The module provides access to ERA5 climate forcing data through the `era5_downloader` package:

```python
from cryogrid_pytools.data import make_era5_downloader

# Create an ERA5 downloader instance
era5 = make_era5_downloader()

# Download ERA5 data for your region and time period
forcing = era5.get_data(
    bbox_WSEN=bbox,
    start_date="2018-01-01",
    end_date="2024-12-31"
)
```

## Advanced Usage

### Smoothing Data

You can smooth any spatial data using a rolling mean filter:

```python
smoothed_dem = cgt.data.smooth_data(
    dem,
    kernel_size=3,
    n_iters=2
)
```

### Working with Sentinel-2 Data

Get raw Sentinel-2 data for custom analysis:

```python
sentinel_data = cgt.data.get_sentinel2_data(
    bbox_WSEN=bbox,
    years=range(2018, 2025),
    assets=['SCL'],  # Scene Classification Layer
    res_m=30,
    epsg=32643,
    max_cloud_cover=5  # Maximum cloud cover percentage
)
```

## Notes

1. All spatial functions support consistent coordinate reference systems through the `epsg` parameter
2. Resolution can be specified in meters using the `res_m` parameter
3. The module handles data downloads and caching automatically
4. Most functions support both vector (GeoDataFrame) and raster (xarray.DataArray) outputs
5. Functions are decorated to handle both bounding box inputs and existing DataArrays for reprojection
