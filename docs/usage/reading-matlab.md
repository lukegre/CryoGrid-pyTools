# Reading MATLAB Structs

CryoGrid-pyTools provides functionality to read MATLAB struct files into Python. Here's how to work with MATLAB data.

## Important Note

!!! warning
    The `run_info.mat` file cannot be read directly as it contains special classes not supported by `scipy.io.loadmat`. You'll need to save the required data in a different format from MATLAB.

## Preparing MATLAB Data

When working in MATLAB, ensure you:

1. Add the `CryoGrid/source` directory to the MATLAB path before saving files
2. Save data in a compatible format

For example, to save parts of `run_info`, use this MATLAB code:

```matlab
% Save specific variables from run_info
save('my_data.mat', 'variable1', 'variable2', '-v7.3')
```

## Reading MATLAB Files in Python

Once you have your MATLAB data in a compatible format:

```python
import cryogrid_pytools as cgt

# Read the MATLAB file
data = cgt.read_matlab_file('my_data.mat')
```

The data will be converted to appropriate Python data structures, making it easy to work with in your Python environment.
