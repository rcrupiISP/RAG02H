from qdrant_client.qdrant_client import QdrantClient
from utils.read_config import get_config_from_path
from retrieval.vdb_wrapper import SearchInVdb
from ingestion.vdb_wrapper import LoadInVdb
from llm.api_call import main_api_call
from ingestion.ingesting import ingest
from functools import partial
from pydantic import BaseModel, ConfigDict
from typing import Callable, Optional
import streamlit as st


class AppParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dct_config: Optional[dict] = None
    vdb_client: Optional[QdrantClient] = None
    searcher: Optional[SearchInVdb] = None
    loader: Optional[LoadInVdb] = None

    ingest: Optional[Callable[[str], None]] = None
    llm_gen_answer: Optional[Callable[[str], str]] = None


@st.experimental_singleton
def initialize() -> AppParams:
    dct_config = get_config_from_path("config.yaml")

    client = QdrantClient(path=dct_config["VECTOR_DB"]["PATH_TO_FOLDER"])

    # for Retrieval
    searcher = SearchInVdb(
        client=client, coll_name=dct_config["VECTOR_DB"]["COLLECTION_NAME"]
    )

    # for Ingestion - indexing
    collection_name = dct_config["VECTOR_DB"]["COLLECTION_NAME"]
    collection_fresh_start = dct_config["VECTOR_DB"]["COLL_FRESH_START"]
    html_folder_path = dct_config["INPUT_DATA"]["PATH_TO_FOLDER"]
    loader = LoadInVdb(client=client, coll_name=collection_name)

    complete_ingest = partial(
        ingest,
        loader=loader,
        is_fresh_start=collection_fresh_start,
        html_folder_path=html_folder_path,
    )

    llm_gen_answer = partial(main_api_call, searcher=searcher)

    out = AppParams(
        dct_config=dct_config,
        vdb_client=client,
        searcher=searcher,
        loader=loader,
        ingest=complete_ingest,
        llm_gen_answer=llm_gen_answer,
    )
    return out
