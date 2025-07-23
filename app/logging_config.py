import logging
import sys
from logging.handlers import TimedRotatingFileHandler

def setup_logging(log_level=logging.INFO):
    log_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s (%(filename)s:%(lineno)d)'
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(stream_handler)

    # File handler (rotates daily)
    file_handler = TimedRotatingFileHandler(
        'app.log', when='midnight', interval=1, backupCount=7
    )
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    return root_logger
