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
    def __init__(self, fname_xls: str, checks=True):
        """
        fname_xls: str
            Path to the CryoGrid Excel configuration file
        checks: bool
            If True, check if the forcing file name matches the given forcing years
        """
        self.fname = pathlib.Path(fname_xls).resolve()
        self.root = self._get_root_path()
        self._df = self._load_xls(fname_xls)
        logger.success(f"Loaded CryoGrid Excel configuration file: {self.fname}")

        if checks:
            self.check_forcing_fname_times()

        self.fname = Munch()
        self.fname.dem = self.get_dem_path()
        self.fname.coords = self.get_coord_path()
        self.fname.era5 = self.get_forcing_path()
        self.fname.datasets = self.get_dataset_paths()
        
        self.time = self.get_start_end_times()

        logger.info(f"Start and end times: {self.time.time_start:%Y-%m-%d} - {self.time.time_end:%Y-%m-%d}")

    def _get_root_path(self):
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
        times = self.get_class('set_start_end_time').T.filter(regex='time')
        times = times.map(lambda x: pd.Timestamp(year=int(x[0]), month=int(x[1]), day=int(x[2])))

        start = times.start_time.min()
        end = times.end_time.max()
        times = pd.Series([start, end], index=['time_start', 'time_end'])

        return times

    def get_coord_path(self):
        fname = self.get_class_filepath('COORDINATES_FROM_FILE', fname_key='file_name', index=1)
        return fname

    def get_dataset_paths(self):
        paths = self.get_class_filepath('READ_DATASET', fname_key='filename').to_frame(name='filepath').T
        
        datasets = self.get_class('READ_DATASET')
        variable = datasets.T.variable_name
        paths.loc['variable'] = variable

        paths = Munch(**paths.T.set_index('variable').filepath.to_dict())
        
        return paths
    
    def get_dem_path(self):
        fname = self.get_class_filepath('DEM', folder_key='folder', fname_key='filename', index=1)
        return fname
    
    def get_forcing_path(self, class_name='read_mat_ERA'):
        fname = self.get_class_filepath(class_name, folder_key='path', fname_key='filename', index=1)
        return fname
    
    def check_forcing_fname_times(self):
        """
        a quick check to see if the file name matches the given forcing years
        """
        import re

        fname = self.get_forcing_path()
        times = self.get_start_end_times().dt.year.astype(str).values.tolist()

        fname_years = re.findall(r'[-_]([12][1089][0-9][0-9])', fname.stem)
        
        assert times == fname_years, f"File name years do not match the forcing years: forcing {times} != fname {fname_years}"
    
    def _load_xls(self, fname_xls: str) -> pd.DataFrame:
        import string

        alph = list(string.ascii_uppercase)
        alphabet_extra = alph + [a+b for a in alph for b in alph]

        df = pd.read_excel(fname_xls, header=None, dtype=str)
        df.columns = [c for c in alphabet_extra[:df.columns.size]]
        df.index = df.index + 1

        return df

    def _get_unique_key(self, key: str, col_value='B'):
        df  = self._df
        idx = df.A == key
        value = df.loc[idx, col_value].values
        if len(value) == 0:
            return None
        elif len(value) > 1:
            raise ValueError(f"Multiple values found for key: {key}")
        else:
            return value[0]
        
    def get_class_filepath(self, key, folder_key='folder', fname_key='file', index=None):

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
        
    def get_class(self, class_name: str):
        df = self._df
        i0s = df.A == class_name
        i0s = i0s[i0s].index.values

        blocks = [self._find_class_block(i0) for i0 in i0s]
        try:
            df = pd.concat(blocks, axis=1)
        except:
            df = blocks
        return df

    def _find_class_block(self, class_idx0: int):
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
    
    def _process_class_block(self, df: pd.DataFrame)->pd.DataFrame:
        """hacky way to process the class block"""
        # drop CLASS_END row
        df = df[df.A != 'CLASS_END']

        # if any cell starts with '>', it is a comment
        df = df.map(lambda x: x if not str(x).startswith('>') else np.nan)

        # drop rows and columns that are all NaN
        df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
        df = df.astype(str)

        # H_LIST and V_MATRIX are special cases
        # return df
        contains_matrix = df.map(lambda x: 'MATRIX' in x).values
        contains_end = df.map(lambda x: 'END' in x).values
        if contains_matrix.any():
            ends = np.where(contains_end)
            r0, c0 = [a[0] for a in np.where(contains_matrix)]

            assert c0 == 1, "Matrix must be in second column"
            assert len(ends) == 2, "Only two ENDs are allowed"
            assert r0 == ends[0][0]
            assert c0 == ends[1][1]

            r1 = ends[0][1]
            c1 = ends[1][0]
            
            arr = df.iloc[r0:r1, c0:c1].values

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