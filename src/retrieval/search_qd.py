from qdrant_client.models import SparseVector, ScoredPoint

from embedding.dense import compute_dense_vector
from embedding.sparse import compute_sparse_vector
from retrieval.vdb_wrapper import SearchInVdb


def print_info(r: ScoredPoint):
    print("ID:", r.id)
    print("SCORE: ", r.score)
    print("PAYLOAD: ", r.payload)
    print()


def main_search(searcher: SearchInVdb, query_text: str):
    query_sparse_vector = SparseVector(**compute_sparse_vector(query_text))
    query_dense_vector = compute_dense_vector(query_text)

    # print("\n\nDense search:")
    # for r in searcher.dense(query_dense_vector, k=5):
    #     print_info(r)

    # print("\n\nSparse search:")
    # for r in searcher.sparse(query_sparse_vector, k=5):
    #     print_info(r)

    # print("\n\nHybrid search:")
    res = searcher.hybrid_qd(
        de_query_vector=query_dense_vector,
        sp_query_vector=query_sparse_vector,
        sp_k=20,
        de_k=20,
        k=5,
    )
    return res


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()

    from qdrant_client.qdrant_client import QdrantClient
    from utils.read_config import get_config_from_path

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
