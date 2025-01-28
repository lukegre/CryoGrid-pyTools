# Installation

CryoGrid-pyTools can be easily installed using pip:

```bash
pip install cryogrid_pytools
```

## Optional Dependencies

CryoGrid-pyTools has several optional dependency groups that can be installed based on your needs:

### Documentation Dependencies
To build the documentation locally, install with the `docs` extra:

```bash
pip install "cryogrid_pytools[docs]"
```

### Data Processing Dependencies
For additional data processing capabilities, install with the `data` extra:

```bash
pip install "cryogrid_pytools[data]"
```

## Development Installation

If you want to contribute to the development of CryoGrid-pyTools, you can install from source:

```bash
git clone https://github.com/lukegre/CryoGrid-pyTools.git
cd CryoGrid-pyTools
pip install -e ".[docs,data]"  # install with all optional dependencies
```

## Requirements

CryoGrid-pyTools requires Python 3.9 or later. The main dependencies are:

- numpy >= 2.0
- scipy >= 1.13.1
- xarray >= 2024
- pandas >= 2
- dask[array,diagnostics] >= 2024

These dependencies will be automatically installed when you install the package using pip.
