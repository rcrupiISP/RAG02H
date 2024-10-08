import os
import pickle
from typing import Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Function to load FAISS index
def load_faiss_index(index_file: str) -> faiss.Index:
    """Loads a FAISS index from a specified file.

    Args:
        index_file (str): The path to the FAISS index file.

    Returns:
        faiss.Index: The loaded FAISS index.
    """
    index = faiss.read_index(index_file)
    return index


# Function to search in the FAISS index
def search_in_faiss(
    index: faiss.Index, query: str, model: SentenceTransformer, k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """Searches for a query in the FAISS index and returns distances and indices of the nearest neighbors.

    Args:
        index (faiss.Index): The FAISS index to search in.
        query (str): The query string to search for.
        model (SentenceTransformer): The model used to generate embeddings for the query.
        k (int): The number of nearest neighbors to return.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the distances and indices of the nearest neighbors.
    """
    # Generate an embedding for the query
    query_embedding = model.encode([query])

    # Perform the search in the FAISS index
    distances, indices = index.search(np.array(query_embedding), k)

    return distances, indices


if __name__ == "__main__":
    # Set the path to the FAISS index
    project_root = os.getenv("MY_HOME", ".")
    faiss_index_file = os.path.join(
        project_root, "embeddings", "faiss_index", "index.faiss"
    )

    # Load the FAISS index
    index = load_faiss_index(faiss_index_file)

    # Load the same SentenceTransformer model used for creating the index
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Get a query from the user
    query = "these particles are accounted to release 70 MeV inside the scintillator"

    # Search in the FAISS index
    distances, indices = search_in_faiss(index, query, model)

    # Print the results
    with open(faiss_index_file + "_pkl", "rb") as file:
        loaded_list_chunk = pickle.load(file)
    print(f"Search results for query: '{query}'")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        print(f"Rank {i + 1}: Chunk Index {idx}, Distance: {dist}")
        print(loaded_list_chunk[idx])
