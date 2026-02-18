import logging
import os
import yaml
import logging.config
from pathlib import Path

def get_logger(name: str):
    logger =  logging.getLogger(name)

    if not logger.hasHandlers():  # Prevent adding handlers multiple times to avaoid dupicates
        config_path = Path("config/logging.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
             # 🔥 CREATE LOG DIRECTORY BEFORE CONFIGURING
        log_file = config["handlers"]["file"]["filename"]
        log_dir = os.path.dirname(log_file)

        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        logging.config.dictConfig(config)

    return logger