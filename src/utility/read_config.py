import os
import logging
import dotenv
from pyaml_env import parse_config

dotenv.load_dotenv()


def get_config_from_path(file_name: str) -> dict:
    try:
        path_to_yaml_files = os.environ["APP_CONF_DIR"]
    except KeyError:
        path_to_yaml_files = './src/config'
        logging.info(f'APP_CONF_DIR not set: searching for config files in: {path_to_yaml_files}')

    full_path = os.path.join(path_to_yaml_files, file_name)

    if full_path.endswith(".yaml"):
        return parse_config(full_path)
    else:
        raise ValueError(f"Only .yaml files are managed.")
