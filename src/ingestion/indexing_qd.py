import os
from logging import getLogger

from qdrant_client import models

from embedding.dense import compute_dense_vector
from embedding.sparse import compute_sparse_vector
from ingestion.utils import chunk_text, convert_html_to_markdown
from ingestion.vdb_wrapper import LoadInVdb

logger = getLogger("ingestion")


def main_indexing(
    loader: LoadInVdb, is_fresh_start: bool, html_folder_path: str
) -> None:
    """
    Indexes HTML files by converting them to markdown and adding the resulting chunks to the vector database.

    Args:
        loader (LoadInVdb): The LoadInVdb instance used to load data into the vector database.
        is_fresh_start (bool): Indicates whether to start fresh with a new collection.
        html_folder_path (str): The path to the folder containing HTML files to be indexed.
    """
    loader.setup_collection(is_fresh_start=is_fresh_start)

    for f in os.listdir(html_folder_path):
        html_file_path = os.path.join(html_folder_path, f)
        if not html_file_path.endswith(".html"):
            logger.info(f"Indexing in vect skipped for file: {html_file_path}")
            continue

        # Convert HTML to markdown
        markdown_text = convert_html_to_markdown(html_file_path)

        # Chunk the Markdown text
        chunks = chunk_text(markdown_text, chunking_mode="markdown_specific")

        # add the chunks to the vector db

        if len(chunks) > 0:
            logger.info(f"Starting indexing in vect db for: {html_file_path}")
            # TODO: more informative payloads might be created during ingestion phase
            loader.add_to_collection(
                dense_vectors=[
                    compute_dense_vector(query_text=chunk["text"]) for chunk in chunks
                ],
                sparse_vectors=[
                    models.SparseVector(**compute_sparse_vector(query_text=chunk["text"]))
                    for chunk in chunks
                ],
                payloads=[{"text": chunk["text"]} for chunk in chunks],
            )
            logger.info(f"Indexing in vect db ended for: {html_file_path}")
        else:
            logger.info(
                f"Indexing in vect db skipped (no chunks) for: {html_file_path}"
            )


if __name__ == "__main__":
    from qdrant_client.qdrant_client import QdrantClient

    from utility.read_config import get_config_from_path
    logger.setLevel('INFO')

    dct_config = get_config_from_path("config.yaml")
    client = QdrantClient(path=dct_config["VECTOR_DB"]["PATH_TO_FOLDER"])

    COLLECTION_NAME = dct_config["VECTOR_DB"]["COLLECTION_NAME"]
    COLL_FRESH_START = dct_config["VECTOR_DB"]["COLL_FRESH_START"]
    html_folder_path = dct_config["INPUT_DATA"]["PATH_TO_FOLDER"]
    loader = LoadInVdb(client=client, coll_name=COLLECTION_NAME)

    main_indexing(
        loader=loader,
        is_fresh_start=COLL_FRESH_START,
        html_folder_path=html_folder_path,
    )
