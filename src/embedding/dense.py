from sentence_transformers import SentenceTransformer

from utility.read_config import get_config_from_path

dct_config = get_config_from_path("config.yaml")

encoder = SentenceTransformer(dct_config["PRE_TRAINED_EMB"]["DENSE_MODEL_NAME"])
EMB_DIM = encoder.get_sentence_embedding_dimension()


def compute_dense_vector(query_text: str) -> list[float]:
    """
    Computes a dense vector representation of the given query text.

    Args:
        query_text (str): The input text to convert into a dense vector.

    Returns:
        list[float]: A list representing the dense vector of the input text.
    """
    return encoder.encode(query_text, show_progress_bar=False).tolist()
