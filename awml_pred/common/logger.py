import logging
from logging import FileHandler, Handler, LogRecord, getLogger

import coloredlogs
from tqdm import tqdm

from awml_pred.typing import Logger

__all__ = ("create_logger",)


class TqdmHandler(Handler):
    def __init__(self, level: int = 0) -> None:
        super().__init__(level)

    def emit(self, record: LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)
            raise


class TqdmFileHandler(FileHandler):
    def __init__(
        self,
        filename: str,
        mode: str = "a",
        encoding: str | None = None,
        *,
        delay: bool = False,
        errors: str | None = None,
    ) -> None:
        super().__init__(filename, mode, encoding, delay, errors)

    def emit(self, record: LogRecord) -> None:
        try:
            super().emit(record)
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)
            raise


def create_logger(
    log_file: str | None = None,
    log_level: int = logging.INFO,
    rank: int = 0,
    modname: str = __name__,
) -> Logger:
    """Return logger.

    Args:
    ----
        log_file (str | None, optional): File path to save log. Defaults to None.
        log_level (int, optional): Level of log. Defaults to logging.INFO.
        rank (int, optional): Rank of current process. Defaults to 0.
        modname (str, optional): Module name. Defaults to __name__.

    Returns:
    -------
        Logger: Created logger.

    """
    logger = getLogger(modname)
    logger.propagate = False
    logger.setLevel(log_level if rank == 0 else logging.ERROR)

    # reset handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    if log_file is not None:
        handler = TqdmFileHandler(filename=log_file)
        handler.setLevel(log_level if rank == 0 else logging.ERROR)
    else:
        handler = TqdmHandler(log_level if rank == 0 else logging.ERROR)

    formatter = coloredlogs.ColoredFormatter(
        fmt="[%(asctime)s][%(processName)s:%(process)s:%(pathname)s.%(funcName)s.%(lineno)d]: [%(levelname)s] %(message)s",  # noqa
        datefmt="%Y-%d-%d %H:%M:%S",
        level_styles={
            "critical": {"color": "red", "bold": True},
            "error": {"color": "red"},
            "warning": {"color": "yellow"},
            "notice": {"color": "magenta"},
            "info": {"color": "white"},
            "debug": {"color": "green"},
            "spam": {"color": "green", "faint": True},
            "success": {"color": "green", "bold": True},
            "verbose": {"color": "blue"},
        },
        field_styles={
            "asctime": {"color": "green"},
            "levelname": {"color": "yellow", "bold": True},
            "processName": {"color": "magenta"},
            "process": {"color": "magenta"},
            "thread": {"color": "blue"},
            "pathname": {"color": "cyan"},
            "funcName": {"color": "blue"},
            "lineno": {"color": "blue", "bold": True},
        },
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
