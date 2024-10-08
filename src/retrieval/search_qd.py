from typing import List

from qdrant_client.models import ScoredPoint, SparseVector

from embedding.dense import compute_dense_vector
from embedding.sparse import compute_sparse_vector
from retrieval.vdb_wrapper import SearchInVdb


def print_info(r: ScoredPoint):
    """
    Prints the information of a scored point.

    Args:
        r (ScoredPoint): The scored point containing id, score, and payload.
    """
    print("ID:", r.id)
    print("SCORE: ", r.score)
    print("PAYLOAD: ", r.payload)
    print()


def main_search(
    searcher: SearchInVdb, query_text: str, sp_k: int = 20, de_k: int = 20, k: int = 5
) -> List[ScoredPoint]:
    """
    Performs a search using the provided searcher with the given query text.

    Args:
        searcher (SearchInVdb): The SearchInVdb instance used to perform the search.
        query_text (str): The query text to be converted into dense and sparse vectors.
        sp_k (int): The number of top results to return from the sparse search.
        de_k (int): The number of top results to return from the dense search.
        k (int): The total number of results to return.

    Returns:
        List[ScoredPoint]: The list of scored points resulting from the search.
    """
    query_sparse_vector = SparseVector(**compute_sparse_vector(query_text))
    query_dense_vector = compute_dense_vector(query_text)

    # res = searcher.dense(query_dense_vector, k=5)
    # res = searcher.sparse(query_sparse_vector, k=5)

    res = searcher.hybrid_qd(
        de_query_vector=query_dense_vector,
        sp_query_vector=query_sparse_vector,
        sp_k=20,
        de_k=20,
        k=5,
    )
    return res


if __name__ == "__main__":
    from qdrant_client.qdrant_client import QdrantClient

    from utility.read_config import get_config_from_path

    dct_config = get_config_from_path("config.yaml")

    client = QdrantClient(path=dct_config["VECTOR_DB"]["PATH_TO_FOLDER"])
    searcher = SearchInVdb(
        client=client, coll_name=dct_config["VECTOR_DB"]["COLLECTION_NAME"]
    )

    # Get a query from the user
    query_text = (
        "these particles are accounted to release 70 MeV inside the scintillator"
    )

    print("Search results:")
    res = main_search(searcher=searcher, query_text=query_text)
    for r in res:
        print_info(r)
