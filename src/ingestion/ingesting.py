from logging import getLogger

from ingestion.download_html import main_html_download
from ingestion.indexing_qd import main_indexing
from ingestion.vdb_wrapper import LoadInVdb

logger = getLogger("ingestion")


def ingest(
    keyword: str,
    loader: LoadInVdb,
    is_fresh_start_dwnld: bool,
    is_fresh_start_indexing: bool,
    html_folder_path: str,
    n_max_docs: int,
) -> None:
    """Downloads documents based on a keyword and indexes them into a vector database.

    Args:
        keyword (str): The keyword to search for and download documents.
        loader (LoadInVdb): An instance of LoadInVdb to handle document indexing.
        is_fresh_start_dwnld (bool): Flag indicating whether to start fresh for downloading.
        is_fresh_start_indexing (bool): Flag indicating whether to start fresh for indexing.
        html_folder_path (str): The directory path where downloaded HTML files will be stored.
        n_max_docs (int): The maximum number of documents to download.
    """
    main_html_download(
        keyword,
        html_folder_path,
        is_fresh_start=is_fresh_start_dwnld,
        n_max_docs=n_max_docs,
    )
    logger.info("Document download ended")

    main_indexing(
        loader=loader,
        is_fresh_start=is_fresh_start_indexing,
        html_folder_path=html_folder_path,
    )
    logger.info("Document indexing ended")


if __name__ == "__main__":
    from qdrant_client.qdrant_client import QdrantClient

    from utility.read_config import get_config_from_path

    dct_config = get_config_from_path("config.yaml")
    client = QdrantClient(path=dct_config["VECTOR_DB"]["PATH_TO_FOLDER"])

    COLLECTION_NAME = dct_config["VECTOR_DB"]["COLLECTION_NAME"]
    COLL_FRESH_START = dct_config["VECTOR_DB"]["COLL_FRESH_START"]
    html_folder_path = dct_config["INPUT_DATA"]["PATH_TO_FOLDER"]
    fresh_start_dwnld = dct_config["INPUT_DATA"]["DOWNLOAD_FRESH_START"]
    n_max_docs = dct_config["INPUT_DATA"]["N_MAX_DOCS"]

    loader = LoadInVdb(client=client, coll_name=COLLECTION_NAME)

    ingest(
        keyword="Riccardo Crupi",
        loader=loader,
        is_fresh_start_dwnld=fresh_start_dwnld,
        is_fresh_start_indexing=COLL_FRESH_START,
        html_folder_path=html_folder_path,
        n_max_docs=n_max_docs,
    )
