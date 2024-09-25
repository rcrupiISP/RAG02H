from qdrant_client.models import SparseVector

from embedding.dense import compute_dense_vector
from embedding.sparse import compute_sparse_vector
from retrieval.vdb_wrapper import SearchInVdb

if __name__ == "__main__":
    from qdrant_client.qdrant_client import QdrantClient

    # TODO: move to config
    PATH_TO_DB = r'c:\profili\u411319\Documents\tmp\qdrant_1'
    client = QdrantClient(path=PATH_TO_DB)

    # TODO: create main with arguments
    COLLECTION_NAME = 'articles'
    COLL_FRESH_START = True
    # query_text

    # Get a query from the user
    query_text = "these particles are accounted to release 70 MeV inside the scintillator"

    query_sparse_vector = SparseVector(**compute_sparse_vector(query_text))
    query_dense_vector = compute_dense_vector(query_text)

    searches = SearchInVdb(client=client, coll_name=COLLECTION_NAME)

    print('Dense search:')
    for r in searches.dense(query_dense_vector, k=5):
        print(r.id, r.score, r.payload)

    print('Sparse search:')
    for r in searches.sparse(query_sparse_vector, k=5):
        print(r.id, r.score, r.payload)

    print('Hybrid search:')
    for r in searches.hybrid_qd(de_query_vector=query_dense_vector,
                                sp_query_vector=query_sparse_vector,
                                sp_k=20, de_k=20, k=5):
        print(r.id, r.score, r.payload)
