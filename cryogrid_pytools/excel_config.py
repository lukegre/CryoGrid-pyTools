# standalone file that can be shared without the rest of the package
import pathlib
import pandas as pd
import numpy as np
from loguru import logger
from munch import Munch


class CryoGridConfigExcel:
    """
    A class to read CryoGrid Excel configuration files and extract file paths
    and maybe in the future do some checks etc
    """
    def __init__(self, fname_xls: str, check_file_paths=True, check_strat_layers=True):
        """
        Initialize the CryoGridConfigExcel object.

        Reads in the Excel configuration file and parse the different classes
        using a pandas DataFrame approach. 

        Parameters
        ----------
        fname_xls : path-like
            Path to the CryoGrid Excel configuration file.
        check_file_paths : bool, default=True, optional
            If True, perform a check that all files linked in the configuration 
            can be found (path exists)
        check_strat_layers : bool, default=True, optional
            If True, perform a check that stratigraphy layer parameters are 
            physically plausible
        """
        self.fname = pathlib.Path(fname_xls).resolve()
        self.root = self._get_root_path()
        self._df = self._load_xls(fname_xls)
        logger.success(f"Loaded CryoGrid Excel configuration file: {self.fname}")

        self.fname = Munch()
        self.fname.dem = self.get_dem_path()
        self.fname.coords = self.get_coord_path()
        self.fname.era5 = self.get_forcing_path()
        self.fname.datasets = self.get_dataset_paths()
        
        self.time = self.get_start_end_times()

        if check_file_paths:
            self.check_files_exist()
        if check_strat_layers:
            self.check_strat_layers()

        logger.info(f"Start and end times: {self.time.time_start:%Y-%m-%d} - {self.time.time_end:%Y-%m-%d}")

    def _get_root_path(self):
        """
        Find and set the root path by locating the 'run_cryogrid.m' file.

        Returns
        -------
        pathlib.Path
            The discovered root path or the current directory if not found.
        """
        path = self.fname.parent
        while True:
            flist = path.glob('run_cryogrid.m')
            if len(list(flist)) > 0:
                self.root = path
                logger.info(f"Found root path: {path}")
                return self.root
            elif str(path) == '/':
                logger.warning("Could not find root path. Set to current directory. You can change this manually with excel_config.root = pathlib.Path('/path/to/root')")
                return pathlib.Path('.')
            else:
                path = path.parent

    def get_start_end_times(self):
        """
        Retrieve the start and end times from the Excel configuration.

        Returns
        -------
        pandas.Series
            A Series with 'time_start' and 'time_end' as Timestamp objects.
        """
        times = self.get_class('set_start_end_time').T.filter(regex='time')
        times = times.map(lambda x: pd.Timestamp(year=int(x[0]), month=int(x[1]), day=int(x[2])))

        start = times.start_time.min()
        end = times.end_time.max()
        times = pd.Series([start, end], index=['time_start', 'time_end'])

        return times

    def get_coord_path(self):
        """
        Get the path to the coordinates file from the Excel configuration.

        Returns
        -------
        pathlib.Path
            The file path for coordinates.
        """
        fname = self.get_class_filepath('COORDINATES_FROM_FILE', fname_key='file_name', index=1)
        return fname

    def get_dataset_paths(self):
        """
        Retrieve paths for each dataset from the Excel configuration.

        Returns
        -------
        munch.Munch
            A dictionary-like object mapping dataset variable names to file paths.
        """
        paths = self.get_class_filepath('READ_DATASET', fname_key='filename').to_frame(name='filepath').T
        
        datasets = self.get_class('READ_DATASET')
        variable = datasets.T.variable_name
        paths.loc['variable'] = variable

        paths = Munch(**paths.T.set_index('variable').filepath.to_dict())
        
        return paths
    
    def get_dem_path(self):
        """
        Get the path for the DEM file from the Excel configuration.

        Returns
        -------
        pathlib.Path
            The DEM file path.
        """
        fname = self.get_class_filepath('DEM', folder_key='folder', fname_key='filename', index=1)
        return fname
    
    def get_forcing_path(self, class_name='read_mat_ERA'):
        """
        Obtain the forcing file path from the Excel configuration.

        Parameters
        ----------
        class_name : str, optional
            The class name to search for in the configuration, by default 'read_mat_ERA'.

        Returns
        -------
        pathlib.Path
            The forcing file path.
        """
        fname = self.get_class_filepath(class_name, folder_key='path', fname_key='filename', index=1)
        return fname
    
    def get_output_max_depth(self, output_class="OUT_regridded_FCI2", depth_key='depth_below_ground')->int:
        """
        Get the maximum depth of the output file from the Excel configuration.

        Parameters
        ----------
        output_class : str, optional
            The class name to search for in the configuration, by default 'OUT_regridded_FCI2'.

        Returns
        -------
        float
            The maximum depth value.
        """
        df = self.get_class(output_class)
        
        depth = str(df.loc[depth_key].iloc[0])
        depth = int(depth)

        return depth

    def check_forcing_fname_times(self):
        """
        Check if the file name matches the forcing years specified in the Excel configuration.

        Raises
        ------
        AssertionError
            If the forcing years in the file name do not match those in the configuration.
        """
        import re

        fname = self.get_forcing_path()
        times = self.get_start_end_times().dt.year.astype(str).values.tolist()

        fname_years = re.findall(r'[-_]([12][1089][0-9][0-9])', fname.stem)
        
        assert times == fname_years, f"File name years do not match the forcing years: forcing {times} != fname {fname_years}"
    
    def _load_xls(self, fname_xls: str) -> pd.DataFrame:
        """
        Load the Excel file into a DataFrame.

        Parameters
        ----------
        fname_xls : str
            Path to the Excel file.

        Returns
        -------
        pandas.DataFrame
            The loaded data with proper indexing and column names.
        """
        import string

        alph = list(string.ascii_uppercase)
        alphabet_extra = alph + [a+b for a in alph for b in alph]

        df = pd.read_excel(fname_xls, header=None, dtype=str)
        df.columns = [c for c in alphabet_extra[:df.columns.size]]
        df.index = df.index + 1

        return df

    def _get_unique_key(self, key: str, col_value='B'):
        """
        Retrieve a single unique value for a given key from the Excel data.

        Parameters
        ----------
        key : str
            The key to look for in column 'A'.
        col_value : str, optional
            The column to retrieve the value from, by default 'B'.

        Returns
        -------
        str or None
            The found value or None if no value exists.

        Raises
        ------
        ValueError
            If multiple values are found for the given key.
        """
        df  = self._df
        idx = df.A == key
        value = df.loc[idx, col_value].values
        if len(value) == 0:
            return None
        elif len(value) > 1:
            raise ValueError(f"Multiple values found for key: {key}")
        else:
            return value[0]
    
    def get_classes(self):
        """
        Returns a dictionary of class names and their corresponding row indices
        from the Excel file. To use as a reference for get_class(<class_name>).

        Returns
        -------
        dict of int: str
            A dictionary mapping class names to row indices
        """
        df = self._df

        class_idx = []
        for i in range(len(df)):
            try: 
                self._find_class_block(i)
                class_idx.append(i)
            except:
                pass

        classes = df.loc[class_idx, 'A'].to_dict()
        return classes

    def get_class_filepath(self, key, folder_key='folder', fname_key='file', index=None):
        """
        Construct a file path from folder and file entries in the Excel configuration.

        Parameters
        ----------
        key : str
            The class name to search for.
        folder_key : str, optional
            Key to identify the folder in the DataFrame, by default 'folder'.
        fname_key : str, optional
            Key to identify the file name in the DataFrame, by default 'file'.
        index : int or None, optional
            If int, return a single entry. Otherwise return all matched entries.

        Returns
        -------
        pathlib.Path or pandas.Series
            The path(s) constructed from the Excel class entries.

        Raises
        ------
        AssertionError
            If multiple folder or filename keys are found.
        TypeError
            If index is not int or None.
        """
        df = self.get_class(key)

        keys = df.index.values
        folder_key = keys[[folder_key in k for k in keys]]
        fname_key = keys[[fname_key in k for k in keys]]

        assert len(folder_key) == 1, f"Multiple folder keys found: {folder_key}"
        assert len(fname_key) == 1, f"Multiple fname keys found: {fname_key}"

        names = df.loc[[folder_key[0], fname_key[0]]]
        names = names.apply(lambda ser: self.root / ser.iloc[0] / ser.iloc[1])

        if index is None:
            return names
        elif isinstance(index, int):
            return names.loc[f"{key}_{index}"]
        else:
            raise TypeError(f"index must be None or int, not {type(index)}")
        
    def get_class(self, class_name: str)->pd.DataFrame:
        """
        Return DataFrame blocks representing the specified class from the Excel data.

        Parameters
        ----------
        class_name : str
            The class name to look up (e.g., 'DEM', 'READ_DATASET').

        Returns
        -------
        pandas.DataFrame
            The concatenated DataFrame of class blocks.
        """
        df = self._df
        i0s = df.A == class_name
        i0s = i0s[i0s].index.values

        blocks = [self._find_class_block(i0) for i0 in i0s]
        try:
            df = pd.concat(blocks, axis=1)
        except:
            # only intended for debugging
            df = blocks  
            logger.warning(f"Could not concatenate blocks for class: {class_name}")
        
        return df

    def _find_class_block(self, class_idx0: int):
        """
        Identify and extract the block of rows corresponding to a class definition.

        Parameters
        ----------
        class_idx0 : int
            Starting row index for the class in the Excel data.

        Returns
        -------
        pandas.DataFrame
            The processed block as a DataFrame.

        Raises
        ------
        AssertionError
            If the class structure is missing required indicators.
        """
        df = self._df

        class_name = df.A.loc[class_idx0]
        msg = f"Given class_idx0 ({class_name}) is not a class. Must have 'index' adjacent or on cell up and right."
        is_index = df.B.loc[class_idx0 - 1: class_idx0].str.contains('index')
        assert is_index.any(), msg

        index_idx = is_index.idxmax()
        class_idx0 = index_idx

        class_idx1 = df.A.loc[class_idx0:] == 'CLASS_END'
        # get first True occurrence
        class_idx1 = class_idx1.idxmax()
        class_block = df.loc[class_idx0:class_idx1]
        class_block = self._process_class_block(class_block)

        return class_block
    
    def _process_class_block(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a raw class block by removing comments, handling special structures, and shaping data.

        Parameters
        ----------
        df : pandas.DataFrame
            A DataFrame slice representing the raw class block.

        Returns
        -------
        pandas.DataFrame
            The cleaned and structured DataFrame of class data.

        Raises
        ------
        AssertionError
            When matrix structures in the block do not match expected format.
        """
        """hacky way to process the class block"""
        # drop CLASS_END row
        df = df[df.A != 'CLASS_END']

        # if any cell starts with '>', it is a comment
        df = df.map(lambda x: x if not str(x).startswith('>') else np.nan)

        # drop rows and columns that are all NaN
        df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
        df = df.astype(str)

        # H_LIST and V_MATRIX are special cases
        contains_matrix = df.map(lambda x: 'MATRIX' in x).values
        contains_vmatrix = df.map(lambda x: 'V_MATRIX' in x).values
        contains_end = df.map(lambda x: 'END' in x).values

        ends = np.where(contains_end)
        if contains_matrix.any():
            r0, c0 = [a[0] for a in np.where(contains_matrix)]

            assert c0 == 1, "Matrix must be in second column"
            assert len(ends) == 2, "Only two ENDs are allowed"
            assert r0 == ends[0][0]
            assert c0 == ends[1][1]

            r1 = ends[0][1]
            c1 = ends[1][0]
            
            arr = df.iloc[r0:r1, c0:c1].values
            if contains_vmatrix.any():
                # first column of V_MATRIX is the index but is not in the config file
                # so we create it. It is one shorter than the num of rows because of header
                arr[1:, 0] = np.arange(r1 - r0 - 1)

            matrix = pd.DataFrame(arr[1:, 1:], index=arr[1:, 0], columns=arr[0, 1:])
            matrix.index.name = matrix.columns.name = df.iloc[r0, 0]
            df = df.drop(index=df.index[r0:r1+1])
            df.loc[r0, 'A'] = matrix.index.name
            df.loc[r0, 'B'] = matrix.to_dict(),
        
        for i, row in df.iterrows():
            # H_LIST first
            if row.str.contains('H_LIST').any():
                r0 = 2
                r1 = row.str.contains('END').argmax()
                df.loc[i, 'B'] = row.iloc[r0:r1].values.tolist()

        class_category = df.A.iloc[0]
        class_type = df.A.iloc[1]
        class_index = df.B.iloc[1]
        col_name = f'{class_type}_{class_index}'

        df = (
            df
            .iloc[2:, :2]
            .rename(columns=dict(B=col_name))
            .set_index('A'))
        df.index.name = class_category

        return df
    
    def check_strat_layers(self):
        """
        Run checks to ensure stratigraphy layers have physically plausible parameter values.
        """
        strat_layers = self.get_class('STRAT_layers')
        logger.info("Checking stratigraphy layers...")
        for layer in strat_layers:
            try:
                check_strat_layer_values(strat_layers[layer].iloc[0])
                logger.success(f"[{layer}]  parameters passed checks")
            except ValueError as error:
                logger.warning(f"[{layer}]  {error}")

    def check_files_exist(self):
        """
        Check if all the files in the configuration exist.
        """

        flist = set([
            self.get_forcing_path(), 
            self.get_dem_path(), 
            self.get_coord_path()] 
            + list(self.get_dataset_paths().values()))
        
        logger.info("Checking file locations...")
        for f in flist:
            if not f.exists():
                logger.warning(f"Cannot find file: {f}")
            else:
                logger.success(f"Located file: {f}")


def check_strat_layer_values(tuple_containing_dict):
    """
    Validate that stratigraphy layer parameters are physically plausible.

    Parameters
    ----------
    tuple_containing_dict : tuple
        A tuple containing a dictionary whose keys represent layer parameters.

    Raises
    ------
    ValueError
        If any parameter check fails for the stratigraphy layers.

    Notes
    -----
    #### Definitions
    - `porosity = 1 - mineral - organic`
    - `airspace = porosity - waterIce`
    - `volume = mineral + organic + waterIce`

    #### Checks
    - `field_capacity < porosity`  :  field capacity is a subset of the porosity
    - `airspace >= 0`  :  cannot have negative airspace
    - `volume <= 1`  :  the sum of mineral, organic, and waterIce cannot exceed 1
    - `waterIce <= porosity`  :  waterIce cannot exceed porosity

    """
    dictionary = tuple_containing_dict[0]
    df = pd.DataFrame(dictionary).astype(float).round(3)

    df['porosity'] = (1 - df.mineral - df.organic).round(3)
    df['airspace'] = (df.porosity - df.waterIce).round(3)
    df['volume'] = (df.mineral + df.organic + df.waterIce).round(3)

    checks = pd.DataFrame()
    checks['field_capacity_lt_porosity'] = df.field_capacity <= df.porosity
    checks['airspace_ge_0'] = df.airspace >= 0
    checks['volume_le_1'] = df.volume <= 1
    checks['waterice_le_porosity'] = df.waterIce <= df.porosity
    checks.index.name = 'layer'

    if not checks.values.all():
        raise ValueError(
            "parameters are not physically plausible. "
            "below are the violations: \n"
            + str(checks.T))
