import os

import dotenv
from pyaml_env import parse_config

dotenv.load_dotenv()


def get_config_from_path(file_name: str) -> dict:
    path_to_yaml_files = os.environ["APP_CONF_DIR"]
    full_path = os.path.join(path_to_yaml_files, file_name)

    if full_path[-5:] == ".yaml":
        return parse_config(full_path)
    else:
        raise ValueError(f"Only .yaml files are managed.")
