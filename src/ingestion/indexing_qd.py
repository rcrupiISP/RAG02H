import os

import markdownify
from bs4 import BeautifulSoup
from qdrant_client import models

from embedding.dense import compute_dense_vector
from embedding.sparse import compute_sparse_vector
from ingestion.vdb_wrapper import LoadInVdb


# Function to read the HTML file and convert it to markdown using markdown-it-py
def convert_html_to_markdown(html_file):
    with open(html_file, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Parse the HTML using BeautifulSoup to clean it
    soup = BeautifulSoup(html_content, 'html.parser')

    # Convert HTML to Markdown
    markdown_content = markdownify.markdownify(str(soup), heading_style="ATX")

    return markdown_content


# Function to chunk the markdown content
def chunk_text(text, chunk_size=300):
    # TODO change asap considering the splitting with \n and a min and max size of the chunk
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks


if __name__ == "__main__":
    from qdrant_client.qdrant_client import QdrantClient

    # TODO: move to config
    PATH_TO_DB = r'c:\profili\u411319\Documents\tmp\qdrant_1'
    client = QdrantClient(path=PATH_TO_DB)

    # TODO: create main_indexing with input:
    # html path dirname
    COLLECTION_NAME = 'articles'
    COLL_FRESH_START = True

    # TODO: move this path to config file, relative to project root
    # Set the path of the HTML file
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    html_folder_path = os.path.join(project_root, 'data', 'docs')

    loader = LoadInVdb(client=client, coll_name=COLLECTION_NAME)
    loader.setup_collection(is_fresh_start=COLL_FRESH_START)

    for f in os.listdir(html_folder_path):
        html_file_path = os.path.join(html_folder_path, f)

        # Convert HTML to markdown
        markdown_text = convert_html_to_markdown(html_file_path)

        # Chunk the Markdown text
        chunks = chunk_text(markdown_text)

        # add the chunks to the vector db
        if len(chunks) > 0:
            # TODO: more informative payloads need to be computed during ingestion phase!
            loader.add_to_collection(
                dense_vectors=[compute_dense_vector(query_text=chunk) for chunk in chunks],
                sparse_vectors=[models.SparseVector(**compute_sparse_vector(query_text=chunk)) for chunk in chunks],
                payloads=[{'text': chunk} for chunk in chunks])
