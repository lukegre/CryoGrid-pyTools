# CryoGrid-pyTools
Tools to read in CryoGrid MATLAB data to Python, includes forcing, outputs, DEMs, etc. 

## Installation

`pip install git+https://github.com/lukegre/CryoGrid-pyTools.git`

## Usage

A very basic example is below, but see `demo.ipynb` for more comprehensive examples.
```python
import cryogrid_pytools as cgt

cgt.read_OUT_regridded_FCI2_file(fname)
```