from ingestion.download_html import main_html_download
from ingestion.indexing_qd import main_indexing
from logging import getLogger
from ingestion.vdb_wrapper import LoadInVdb

logger = getLogger("ingestion")


def ingest(
    keyword: str, loader: LoadInVdb, is_fresh_start: bool, html_folder_path: str
):  # TODO: cambiare nome is_fres_start se non è riferito anche a download html
    main_html_download(keyword, html_folder_path)
    logger.info("Doc download ended")
    main_indexing(
        loader=loader,
        is_fresh_start=is_fresh_start,
        html_folder_path=html_folder_path,
    )
    logger.info("Doc indexing ended")


if __name__ == "__main__":
    from qdrant_client.qdrant_client import QdrantClient
    from utils.read_config import get_config_from_path

    dct_config = get_config_from_path("config.yaml")
    client = QdrantClient(path=dct_config["VECTOR_DB"]["PATH_TO_FOLDER"])

    # TODO: create main_indexing with input:
    COLLECTION_NAME = dct_config["VECTOR_DB"]["COLLECTION_NAME"]
    COLL_FRESH_START = dct_config["VECTOR_DB"]["COLL_FRESH_START"]
    html_folder_path = dct_config["INPUT_DATA"]["PATH_TO_FOLDER"]
    loader = LoadInVdb(client=client, coll_name=COLLECTION_NAME)

    ingest(
        keyword="Riccardo Crupi",
        loader=loader,
        is_fresh_start=COLL_FRESH_START,
        html_folder_path=html_folder_path,
    )
