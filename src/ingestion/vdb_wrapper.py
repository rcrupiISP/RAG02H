from typing import Union
from uuid import uuid4

from qdrant_client import QdrantClient, models

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
        Initializes the LoadInVdb instance.

        Args:
            client (QdrantClient): Client to connect to the vector database.
            coll_name (str): Name of the collection.
            dense_vect_name (str): Name of the dense vector.
            sparse_vect_name (str): Name of the sparse vector.
        """
        self.client = client
        self.coll_name = coll_name
        self.dense_vect_name = dense_vect_name
        self.sparse_vect_name = sparse_vect_name

    def setup_collection(self, is_fresh_start: bool = False) -> None:
        """
        Ensures that the collection exists; creates it if it does not.

        Args:
            is_fresh_start (bool): If True, removes the existing collection before re-creation.

        Returns:
            None
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
        ids: Union[list[str], None] = None,
    ) -> None:
        """Adds dense and sparse vectors along with payloads to the collection.

        Args:
            dense_vectors (list[list[float]]): list of dense vectors to add.
            sparse_vectors (list[models.SparseVector]): list of sparse vectors to add.
            payloads (list[dict]): list of payload dictionaries to associate with the vectors.
            ids (Union[list[str], None]): Optional list of IDs for the points. If None, new UUIDs are generated.

        Raises:
            ValueError: If the lengths of the lists do not match.

        Returns:
            None
        """
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
