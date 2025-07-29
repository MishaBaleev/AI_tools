import logging 
from logging import StreamHandler
import sys

class CustomFormatter(logging.Formatter):
    grey = "\033[37m"
    green = "\033[32m"
    yellow = "\033[33m"
    red = "\033[31m"
    bold_red = "\033[31m\033[1m\033[6m"
    reset = "\033[0m"

    FORMATS = {
        logging.DEBUG: "[%(asctime)s]:[" + grey + "%(levelname)s" + reset + "]:["+yellow+"%(name)s"+reset+"]:%(message)s",
        logging.INFO: "[%(asctime)s]:[" + green + "%(levelname)s" + reset + "]:["+yellow+"%(name)s"+reset+"]:%(message)s",
        logging.WARNING: "[%(asctime)s]:[" + yellow + "%(levelname)s" + reset + "]:["+yellow+"%(name)s"+reset+"]:%(message)s",
        logging.ERROR: "[%(asctime)s]:[" + red + "%(levelname)s" + reset + "]:["+yellow+"%(name)s"+reset+"]:%(message)s",
        logging.CRITICAL: "[%(asctime)s]:[" + bold_red + "%(levelname)s" + reset + "]:["+yellow+"%(name)s"+reset+"]:%(message)s"
    }

    def format(self, record):
        import platform
        if platform.system() == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
    
class Logger():
    def __init__(self, logger_name:str):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        handler = StreamHandler(stream=sys.stdout)
        handler.setFormatter(CustomFormatter())
        self.logger.addHandler(handler)

    def disable(self):
        self.logger.handlers.clear()