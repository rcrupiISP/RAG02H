import os
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ingestion.utils import chunk_text, convert_html_to_markdown


# Function to save embeddings to FAISS index
def save_chunks_to_faiss(chunks, index_file):
    # Load a pre-trained transformer model for embedding generation
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Generate embeddings for each chunk
    embeddings = model.encode(chunks)

    # Create a FAISS index and add embeddings
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance index
    index.add(np.array(embeddings))

    # Save the FAISS index to file
    faiss.write_index(index, index_file)
    print(f"FAISS index saved to {index_file}")

    # Function to save chunks to a text file
    with open(index_file + "_pkl", "wb") as file:
        pickle.dump(chunks, file)


if __name__ == "__main__":
    # example of loading one file in faiss index
    project_root = os.getenv("MY_HOME", ".")

    # Set the path of the HTML file
    html_file_path = os.path.join(project_root, "data", "docs", "2401.02900v1.html")

    # Set the path to save FAISS index
    faiss_index_file = os.path.join(
        project_root, "embeddings", "faiss_index", "index.faiss"
    )

    # Convert HTML to markdown
    markdown_text = convert_html_to_markdown(html_file_path)

    # Chunk the Markdown text
    chunks = chunk_text(markdown_text)

    # Save the chunks to FAISS index
    save_chunks_to_faiss(chunks, faiss_index_file)
