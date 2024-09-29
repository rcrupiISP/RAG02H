import logging
from typing import Any, List, Dict, Optional
from pyaml_env import parse_config
import os


def get_config_from_path(file_name: str) -> dict:

    if os.environ.get("MY_HOME") is None:
        tmp_path = os.getcwd()
        rag_index = tmp_path.find('RAG02H')
        root_path = tmp_path[:rag_index + len('RAG02H')]
        os.environ["MY_HOME"] = root_path.replace('\\', '/')
    if os.environ.get("APP_CONF_DIR") is None:
        os.environ["APP_CONF_DIR"] = os.path.join(os.environ["MY_HOME"], "src\\config").replace('\\', '/')

    path_to_yaml_files = os.environ["APP_CONF_DIR"]

    full_path = os.path.join(path_to_yaml_files, file_name)

    if full_path[-5:] == ".yaml":
        return parse_config(full_path)
    else:
        raise ValueError(f"Only .yaml files are managed.")
