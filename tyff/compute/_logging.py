import logging


def _set_up_logger(file_name: str, log_level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("tyff")
    logger.handlers.clear()
    logger.setLevel(log_level)
    handler = logging.FileHandler(file_name)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

    return logger
