import sys

from loguru import logger

from config import LOGS_DIR


def setup_logging():
    logger.remove()

    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    logger.add(
        LOGS_DIR / "app.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="1 week",
        compression="zip",
        backtrace=True,
        diagnose=True,
    )

    logger.add(
        LOGS_DIR / "errors.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message} | {exception}",
        rotation="5 MB",
        retention="2 weeks",
        compression="zip",
        backtrace=True,
        diagnose=True,
    )

    logger.info("Logging system initialized")

    return logger
