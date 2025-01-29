# Working with Forcing Data

CryoGrid-pyTools provides functionality to work with ERA5 forcing data for CryoGrid simulations.

## Reading ERA5 Data

You can read ERA5 forcing data from MATLAB files using the `read_mat_ear5` function:

```python
import cryogrid_pytools as cgt

# Read ERA5 forcing data
ds = cgt.read_mat_ear5('path/to/ERA5.mat')
```

The returned xarray Dataset contains all ERA5 forcing variables needed for CryoGrid simulations.

## Converting ERA5 Data

If you have ERA5 data from the Copernicus Climate Data Store (CDS) in netCDF format, you can convert it to the format expected by CryoGrid using `era5_to_matlab`:

```python
import cryogrid_pytools as cgt

# Convert ERA5 netCDF to MATLAB format
ds = cgt.era5_to_matlab(
    era5_dataset,  # xarray Dataset from ERA5 CDS
    save_path='ERA5.mat'  # Optional: save to MATLAB file
)
```

The function expects the following variables in the input dataset:

### Single Level Variables
- u10, v10: 10m wind components
- sp: Surface pressure
- d2m: 2m dewpoint temperature
- t2m: 2m temperature
- ssrd: Surface solar radiation downwards
- strd: Surface thermal radiation downwards
- tisr: TOA incident solar radiation
- tp: Total precipitation
- Zs: Surface geopotential (static)

### Pressure Level Variables
- t: Temperature
- z: Geopotential
- q: Specific humidity
- u, v: Wind components

The output will be formatted to match the requirements of the `CryoGrid.POST_PROC.read_mat_ERA` class in MATLAB.
