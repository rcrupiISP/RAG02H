import os

import markdownify
from bs4 import BeautifulSoup
from qdrant_client import models

from embedding.dense import compute_dense_vector
from embedding.sparse import compute_sparse_vector
from ingestion.vdb_wrapper import LoadInVdb
from logging import getLogger

logger = getLogger("ingestion")


# Function to read the HTML file and convert it to markdown using markdown-it-py
def convert_html_to_markdown(html_file):
    with open(html_file, "r", encoding="utf-8") as file:
        html_content = file.read()

    # Parse the HTML using BeautifulSoup to clean it
    soup = BeautifulSoup(html_content, "html.parser")

    # Convert HTML to Markdown
    markdown_content = markdownify.markdownify(str(soup), heading_style="ATX")

    return markdown_content


# Function to chunk the markdown content
def chunk_text(text, chunk_size=300):
    # TODO change asap considering the splitting with \n and a min and max size of the chunk
    words = text.split()
    chunks = [
        " ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)
    ]
    return chunks


def main_indexing(loader: LoadInVdb, is_fresh_start: bool, html_folder_path: str):
    loader.setup_collection(is_fresh_start=is_fresh_start)

    for f in os.listdir(html_folder_path):
        html_file_path = os.path.join(html_folder_path, f)
        if not html_file_path.endswith(".html"):
            logger.info(f"Indexing in vect skipped for file: {html_file_path}")
            continue

        # Convert HTML to markdown
        markdown_text = convert_html_to_markdown(html_file_path)

        # Chunk the Markdown text
        chunks = chunk_text(markdown_text)

        # add the chunks to the vector db
        if len(chunks) > 0:
            # TODO: more informative payloads need to be computed during ingestion phase!
            loader.add_to_collection(
                dense_vectors=[
                    compute_dense_vector(query_text=chunk) for chunk in chunks
                ],
                sparse_vectors=[
                    models.SparseVector(**compute_sparse_vector(query_text=chunk))
                    for chunk in chunks
                ],
                payloads=[{"text": chunk} for chunk in chunks],
            )
        logger.info(f"Indexing in vect db ended for: {html_file_path}")


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

    main_indexing(
        loader=loader,
        is_fresh_start=COLL_FRESH_START,
        html_folder_path=html_folder_path,
    )
