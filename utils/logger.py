import logging

def get_logger(stream_handler = True):
    logger = logging.getLogger(name='JED_MAP_Langevin§')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(asctime)s [%(name)s] >> %(message)s")
    if stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    else:
        fh = logging.FileHandler('logging_file.log')
        logger.addHandler(fh)

    return logger