# CryoGrid-pyTools
Tools to read in CryoGrid MATLAB data to Python, includes forcing, outputs, DEMs, etc. 

Feel free to use, modify, and distribute as you see fit.

## Installation

```bash
pip install cryogrid_pytools
```

## Usage
This package can read in 
- CryoGrid output (from `OUT_regridded_FCI2`)
- simple MATLAB struct files (not `run_info.mat`)
- ERA5 forcing
- excel run configuration files (experimental)

Basic examples are shown below for the first two items, but see `demo.ipynb` for more comprehensive examples.

### Reading single CryoGrid output files

Currently, only output files from the calss `OUT_regridded_FCI2` are supported. 
However, the majority of files you'll use follow this format. 
```python
import cryogrid_pytools as cgt

# a single file that should look something like this
fname = "results_from_RUN_SIMPLE/<project_name>_<YYYYMMDD>.mat"

# currently the only output from OUT_regridded_FCI2 is supported
dataset = cgt.read_OUT_regridded_FCI2_file(fname)
```

### Reading multiple CryoGrid output files

If you're working with RUN_SPATIAL_SPINUP_CLUSTERING, you can read in multiple files at once. The data are returned as an `xarray.Dataset` with the dimensions: [`gridcell`, `depth`, `time`]
```python
# path contains wildcards where cluster ID and date are respectively
fname = "results_from_RUN_SPATIAL_SPINUP_CLUSTERING/project_name_*_*.mat"

# deepest_point is the deepest point in the saved files
dataset = cgt.read_OUT_regridded_FCI2_clusters(fname, deepest_point=-5)
```

### Reading struct files from MATLAB
<div style="border: 1px solid; border-radius: 5px; padding: 10px; margin: 10px 0; width: 97.5%; border-color: #0096C7; background-color: rgba(0, 150, 199, 0.1); "><b>TIP</b>: In <code>MATLAB</code>, You need to add the `CryoGrid/source` directory to the MATLAB path before the file can be loaded properly. </div>

Note that `run_info.mat` cannot be read as it contains special classes that are not supported by `scipy.io.loadmat`.  To save parts of `run_info`, use the following code in MATLAB:

```matlab
run_info_spatial = run_info.SPATIAL.STATVAR
save "run_info_spatial.mat" run_info_spatial
```

Then, in Python:

```python
fname = 'path_to_file/run_spatial_statvar.mat'
spatial = cgt.read_mat_struct_as_dataset(fname, index=['Y', 'X'])
```

