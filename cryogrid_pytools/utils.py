

def change_logger_level(level):
    """
    Change the logger level of the cryogrid_pytools logger.

    Parameters
    ----------
    level : str
        Level to change the logger to. Must be one of ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'].
    """
    import sys
    from loguru import logger

    if level in ['INFO', 'WARNING', 'ERROR', 'CRITICAL', 'SUCCESS']:
        format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> - <level>{message}</level>"
    else:
        format = '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>'
    
    
    logger.remove()
    logger.add(sys.stdout, level=level, format=format)



def drop_coords_without_dim(da):
    """
    Drop coordinates that do not have a corresponding dimension.

    Parameters
    ----------
    da : xarray.DataArray
        The input data array.

    Returns
    -------
    xarray.DataArray
        The data array with dropped coordinates.
    """
    for c in da.coords:
        if c not in da.dims:
            da = da.drop_vars(c)
    return da
