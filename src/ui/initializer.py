import logging
import sys
from functools import partial
from logging import getLogger
from typing import Callable, Optional

import streamlit as st
from pydantic import BaseModel, ConfigDict
from qdrant_client.qdrant_client import QdrantClient

from ingestion.ingesting import ingest
from ingestion.vdb_wrapper import LoadInVdb
from llm.api_call import main_api_call
from retrieval.vdb_wrapper import SearchInVdb
from ui.utils import setup_logger as _setup_logger
from utility.read_config import get_config_from_path


class AppParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dct_config: Optional[dict] = None
    vdb_client: Optional[QdrantClient] = None
    searcher: Optional[SearchInVdb] = None
    loader: Optional[LoadInVdb] = None
    log_formatter: Optional[logging.Formatter] = None

    ingest: Optional[Callable[[str], None]] = None
    llm_gen_answer: Optional[Callable[[str], str]] = None
    setup_task_logger: Optional[Callable[[list[logging.Handler]], None]] = None


@st.experimental_singleton
def initialize() -> AppParams:
    dct_config = get_config_from_path("config.yaml")

    log_formatter = logging.Formatter(dct_config["UI"]["APP_LOG_FORMAT"])

    task_logger = getLogger("ingestion")
    setup_task_logger = partial(
        _setup_logger,
        logger=task_logger,
        level=dct_config["UI"]["APP_LOG_LEVEL"],
        propagate=False,
    )

    logging.basicConfig(
        level=dct_config["UI"]["APP_LOG_LEVEL"],
        format=dct_config["UI"]["APP_LOG_FORMAT"],
        force=True,
    )

    client = QdrantClient(path=dct_config["VECTOR_DB"]["PATH_TO_FOLDER"])

    # for Retrieval
    searcher = SearchInVdb(
        client=client, coll_name=dct_config["VECTOR_DB"]["COLLECTION_NAME"]
    )

    # for Ingestion - indexing
    collection_name = dct_config["VECTOR_DB"]["COLLECTION_NAME"]
    collection_fresh_start = dct_config["VECTOR_DB"]["COLL_FRESH_START"]
    html_folder_path = dct_config["INPUT_DATA"]["PATH_TO_FOLDER"]
    dwnld_fresh_start = dct_config["INPUT_DATA"]["DOWNLOAD_FRESH_START"]
    n_max_docs = dct_config["INPUT_DATA"]["N_MAX_DOCS"]
    loader = LoadInVdb(client=client, coll_name=collection_name)

    complete_ingest = partial(
        ingest,
        loader=loader,
        is_fresh_start_dwnld=dwnld_fresh_start,
        is_fresh_start_indexing=collection_fresh_start,
        html_folder_path=html_folder_path,
        n_max_docs=n_max_docs,
    )

    llm_gen_answer = partial(
        main_api_call, searcher=searcher, rewriting=dct_config["RAG"]["QUERY_REWRITING"]
    )

    out = AppParams(
        dct_config=dct_config,
        vdb_client=client,
        searcher=searcher,
        loader=loader,
        log_formatter=log_formatter,
        ingest=complete_ingest,
        llm_gen_answer=llm_gen_answer,
        setup_task_logger=setup_task_logger,
    )

    return out


@st.experimental_singleton
def customize():
    # patch from https://github.com/streamlit/streamlit/issues/3426
    def set_global_exception_handler(f):
        script_runner = sys.modules["streamlit.runtime.scriptrunner.script_runner"]
        script_runner.handle_uncaught_app_exception.__code__ = f.__code__

    def exception_handler(e):
        st.error(f"Oops, an internal error occurred!", icon="😿")
        raise Exception("Exception found!") from e

    set_global_exception_handler(exception_handler)
