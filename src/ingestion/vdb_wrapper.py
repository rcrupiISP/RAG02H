from typing import Union
from uuid import uuid4

from qdrant_client import models, QdrantClient

from embedding.dense import EMB_DIM


class LoadInVdb:
    def __init__(
        self,
        client: QdrantClient,
        coll_name: str,
        dense_vect_name: str = "text-dense",
        sparse_vect_name: str = "text-sparse",
    ):
        """
        :param client: client to connect to vector db
        :param coll_name: name of collection
        :param dense_vect_name: name of dense vector
        :param sparse_vect_name: name of sparse vector
        """
        self.client = client
        self.coll_name = coll_name
        self.dense_vect_name = dense_vect_name
        self.sparse_vect_name = sparse_vect_name

    def setup_collection(self, is_fresh_start: bool = False):
        """
        Makes sure that collection with name coll_name exists, by creating it if it does not exist.
        If is_fresh_start is True, in case a collection with that name already exists, it is removed
        before re-creation.
        :param is_fresh_start: bool, if collection needs to be created anew.
        :return: None
        """
        if is_fresh_start:
            self.client.delete_collection(collection_name=self.coll_name)

        if not self.client.collection_exists(self.coll_name):
            self.client.create_collection(
                collection_name=self.coll_name,
                vectors_config={
                    "text-dense": models.VectorParams(
                        size=EMB_DIM,  # Vector size is defined by used model
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "text-sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False,
                        )
                    )
                },
            )

    def add_to_collection(
        self,
        dense_vectors: list[list[float]],
        sparse_vectors: list[models.SparseVector],
        payloads: list[dict],
        ids: Union[list, None] = None,
    ):
        ids = [str(uuid4()) for _ in dense_vectors] if ids is None else ids
        if not (len(dense_vectors) == len(sparse_vectors) == len(payloads) == len(ids)):
            raise ValueError(
                "ids, dense vector, sparse vector and payloads lists must have the same length"
            )

        self.client.upload_points(
            collection_name=self.coll_name,
            points=[
                models.PointStruct(
                    id=a_id,
                    vector={"text-dense": dense_vector, "text-sparse": sparse_vector},
                    payload=payload,
                )
                for a_id, dense_vector, sparse_vector, payload in zip(
                    ids, dense_vectors, sparse_vectors, payloads
                )
            ],
            max_retries=3,
        )
        print(
            f"Number of elements in {self.coll_name}: {self.client.count(collection_name=self.coll_name)}"
        )
