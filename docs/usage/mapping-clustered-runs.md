# Mapping profile metrics to spatial maps
The key goal here is to map a single value from each spatial run to a map. The key thing to remember is that our profiles represent the temperature at the cluster centroid locations. We would like to map these centroid data to the entire spatial coverage of the cluster. To do this we have to read in the spatial data that contains information about the locations of the cluster centroids. This information can be retrieved from the `run_parameters.mat` file in MATLAB using the script in the [last section](#saving-spatial-data-from-matlab).

## Read the profile data
Using the standard function to read the regridded profiles. The deepest point is set to -20m (taken from the Excel config file).
```python
import cryogrid_pytools as cg

# using regex pattern - [0-9]+ is the run number, 197[5-9] is 1975-1979, [0-9]{4} = mmdd part of the date
fname_profiles = '.../runs/abramov-small-3classes/outputs/abramov-small-3classes_[0-9]+_197[5-9][0-9]{4}.mat'

# the outputs of ds_profiles have dimensions [gridcell, depth, time]
ds_profiles = cg.read_OUT_regridded_clusters(fname_profiles, deepest_point=-20)
```

## Reduce variable to one value per profile
If we want to map the data spatially, we can only do this if we have one aggregated value per profile. For example, we can calculate the mean temperature profiles imported above (1975-1979) and then select the temperature at 2m depth.

```python
# calculate mean temperature profile for each gridcell [gridcell, depth]
temp_profile_mean = (
    ds_profiles['T']  # select the temperature data
    .assign_attrs(long_name='Temperature', units='°C')  # assign attributes to the data (for plotting)
    .mean('time', keep_attrs=True))  # calculate the mean over time, keeping the attributes

temp_2m_mean = (
    temp_profile_mean
    .sel(depth=-2, method='nearest')  # select the temperature at 2m depth (or the nearest depth)
    .drop_vars(['depth', 'elevation'])  # drop the depth and elevation coordinates
    .compute())  # load the data into memory
)
```

## Mapping with spatial data
For this to work, you need to have completed [this part](#saving-spatial-data-from-matlab).
The `cg.spatial_clusters.read_spatial_data` function reads the spatial data from the file saved in MATLAB and does some neat things in the backend. It maps the cluster centroid gridcell numbers to the map, making it easier to map the data to the spatial grid.

```python
fname_spatial = ".../runs/abramov-small-3classes/outputs/run_spatial_info.mat"

ds_spatial = cg.spatial_clusters.read_spatial_data(fname_spatial)

# Here, we map the temperature data to the spatial grid using the cluster centroid gridcell numbers
temp_2m_mapped = cg_spatial.map_gridcells_to_clusters(
    temp_2m_mean,  # dims = [gridcell]
    ds_spatial.cluster_centroid_gridcell_2d,  # dims = [y, x]
)

# plot the output
img = temp_2m_mapped.plot.imshow(robust=True, aspect=2, size=5)
img.axes.set_aspect('equal')
```
![](../imgs/mapped_temp_2m-1975_1979.png)


## Saving spatial data from MATLAB
We need to save spatial data from `run_parameters.mat` to a file that Python can read. The `run_parameters.mat` is not readable by Python because it is a struct array that contains custom CryoGrid classes that Python does not understand. So, to do this one has to run the following code in MATLAB:

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

If you are using `CryoGrid-run-manager` this is automatically done for you by default.
