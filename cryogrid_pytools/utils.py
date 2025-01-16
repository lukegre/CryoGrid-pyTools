

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
