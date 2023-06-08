import logging
import colorlog
from shared.constants import SERVER, ServerType

from shared.logging.constants import LoggingMode, LoggingPayload, LoggingType

class AppLogger(logging.Logger):
    def __init__(self, name='app_logger', log_file=None, log_level=logging.DEBUG):
        super().__init__(name, log_level)
        self.log_file = log_file

        self._configure_logging()

    def _configure_logging(self):
        log_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(levelname)s:%(name)s:%(message)s',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            reset=True,
            secondary_log_colors={},
            style='%'
        )
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(log_formatter)
            self.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        self.addHandler(console_handler)

        # setting logging mode
        if SERVER != ServerType.PRODUCTION.value:
            self.logging_mode = LoggingMode.OFFLINE.value
        else:
            self.logging_mode = LoggingMode.ONLINE.value

    # def log(self, log_type: LoggingType, log_payload: LoggingPayload):
    #     if log_type == LoggingType.DEBUG:
    #         self.debug(log_payload.message)
    #     elif log_type == LoggingType.INFO:
    #         self.info(log_payload.message)
    #     elif log_type == LoggingType.ERROR:
    #         self.error(log_payload.message)
    #     elif log_type in [LoggingType.INFERENCE_CALL, LoggingType.INFERENCE_RESULT]:
    #         self.info(log_payload.message)

    def log(self, log_type: LoggingType, log_message, log_data = None):
        if log_type == LoggingType.DEBUG:
            self.debug(log_message)
        elif log_type == LoggingType.INFO:
            self.info(log_message)
        elif log_type == LoggingType.ERROR:
            self.error(log_message)
        elif log_type in [LoggingType.INFERENCE_CALL, LoggingType.INFERENCE_RESULT]:
            self.info(log_message)
    