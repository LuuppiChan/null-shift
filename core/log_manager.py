import logging

from core.config import CoreConfig, manager


class LogManager:
    """Handles logging settings."""

    def __init__(self) -> None:
        cfg = manager.get_config()
        self.current_level = cfg.log.level
        self.current_silenced = cfg.log.silenced_libraries
        self.current_log_path = cfg.log.file_path
        self.log_to_file = cfg.log.to_file
        self._formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s", datefmt="%H:%M:%S"
        )
        self._init_logging()
        self._reload_logging(cfg)
        self._reload_log_destination(cfg)
        manager.config_updated.connect(self._config_updated)

    def _init_logging(self):
        logging.basicConfig(level=self.current_level.upper())

    def _reload_logging(self, config: CoreConfig):
        """Force reload logging."""
        new_level = config.log.level.upper()
        if hasattr(logging, new_level):
            logging.getLogger().setLevel(new_level)
        else:
            logger.warning("Invalid log level '%s', keeping previous.", new_level)

        for silenced in self.current_silenced:
            logging.getLogger(silenced).setLevel(logging.NOTSET)

        for silenced in config.log.silenced_libraries:
            logging.getLogger(silenced).setLevel(logging.WARNING)

    def _reload_log_destination(self, config: CoreConfig):
        """Force reload log path related logging."""
        root = logging.getLogger()

        for handler in root.handlers[:]:
            handler.close()
            root.removeHandler(handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self._formatter)
        root.addHandler(console_handler)

        if config.log.to_file:
            file_handler = logging.FileHandler(config.log.file_path)
            file_handler.setFormatter(self._formatter)
            root.addHandler(file_handler)

    def _config_updated(self, config: CoreConfig):
        """Signal trigger for checking new config."""
        silenced_changed = self.current_silenced != config.log.silenced_libraries
        level_changed = self.current_level != config.log.level

        if silenced_changed or level_changed:
            self._reload_logging(config)
            self.current_level = config.log.level
            self.current_silenced = config.log.silenced_libraries
            logger.info("Logging system config updated")

        log_to_file_changed = config.log.to_file != self.log_to_file
        file_path_changed = config.log.file_path != self.current_log_path

        if log_to_file_changed or file_path_changed:
            self._reload_log_destination(config)
            self.current_log_path = config.log.file_path
            self.log_to_file = config.log.to_file
            logger.info("Logging path config updated")


log_manager = LogManager()

logger = logging.getLogger(__name__)
