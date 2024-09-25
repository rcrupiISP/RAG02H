from qdrant_client import models, QdrantClient


class SearchInVdb:
    def __init__(self, client: QdrantClient, coll_name: str, dense_vect_name: str = 'text-dense',
                 sparse_vect_name: str = 'text-sparse'):
        self.client = client
        self.coll_name = coll_name
        self.dense_vect_name = dense_vect_name
        self.sparse_vect_name = sparse_vect_name

    def dense(self, query_vector: list[float], k: int = 5) -> list[models.ScoredPoint]:
        hits = \
            self.client.search(
                collection_name=self.coll_name,
                query_vector=models.NamedVector(
                    name=self.dense_vect_name,
                    vector=query_vector,
                ),
                # query_filter=models.Filter(must=[models.FieldCondition(key='title',
                #                                                        match=models.MatchAny(
                #                                                            any=['Super_Bowl_50', 'Orso']))]),
                # many types of filter available (still not tried) among which:
                # range, is Null, exact match, etc..
                # for more info, https://qdrant.tech/articles/vector-search-filtering/
                limit=k,
            )
        return hits

    def sparse(self, query_vector: models.SparseVector, k: int = 5) -> list[models.ScoredPoint]:
        hits = \
            self.client.search(
                collection_name=self.coll_name,
                query_vector=models.NamedSparseVector(
                    name=self.sparse_vect_name,
                    vector=query_vector,
                ),
                limit=k,
            )
        return hits

    def hybrid_qd(self,
                  de_query_vector: list[float],
                  sp_query_vector: models.SparseVector,
                  sp_k: int = 20,
                  de_k: int = 20,
                  k: int = 5):
        # https://qdrant.tech/documentation/concepts/hybrid-queries/#hybrid-search
        hits = \
            self.client.query_points(
                collection_name=self.coll_name,
                prefetch=[
                    models.Prefetch(
                        query=sp_query_vector,
                        using=self.sparse_vect_name,
                        limit=sp_k,
                    ),
                    models.Prefetch(
                        query=de_query_vector,
                        using=self.dense_vect_name,
                        limit=de_k,
                    )
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=k
            ).points
        return hits
