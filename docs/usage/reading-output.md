# Reading CryoGrid Output

CryoGrid-pyTools provides functionality to read CryoGrid output files into Python. Currently, it supports reading regridded FCI2 output files.

## Reading FCI2 Output Files

Use the `read_OUT_regridded_FCI2_file` function to read a regridded FCI2 output file:

```python
import cryogrid_pytools as cgt

# Read a single output file
ds = cgt.read_OUT_regridded_FCI2_file(
    'path/to/your/output_file.mat',
    deepest_point=-5  # Set the deepest point in meters
)
```

The function returns an xarray Dataset with the following variables:
- `time`: Time coordinate (datetime64)
- `depth`: Depth coordinate in meters
- `T`: Temperature
- `water`: Water content
- `ice`: Ice content
- `class_number`: Class number
- `FCI`: Frozen/Thawed state
- `elevation`: Surface elevation

## Reading Multiple Files

For spatial runs with multiple output files, use `read_OUT_regridded_FCI2_clusters`:

```python
# Read multiple output files
ds = cgt.read_OUT_regridded_FCI2_clusters(
    'path/to/output/directory/*.mat',  # Glob pattern for output files
    deepest_point=-5
)
```

The resulting Dataset will have an additional `gridcell` dimension for the spatial component.

## Data Structure

The output is an xarray Dataset with:

- Dimensions:
  - `time`: Timesteps in the simulation
  - `depth`: Vertical grid points
  - `gridcell`: (Only for cluster runs) Spatial grid points

- Variables:
  - All variables are stored as dask arrays for efficient memory usage
  - Temperature and other fields are stored with dimensions (time, depth) or (gridcell, depth, time)

## Working with the Data

Being an xarray Dataset, you can use all standard xarray operations:

```python
# Select a specific time
ds.sel(time='2000-01-01')

# Get mean temperature over time
ds.T.mean(dim='time')

# Plot temperature profile
ds.T.plot()
