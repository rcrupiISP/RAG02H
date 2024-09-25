from sentence_transformers import SentenceTransformer
from utils.read_config import get_config_from_path

dct_config = get_config_from_path("config.yaml")

encoder = SentenceTransformer(dct_config["PRE_TRAINED_EMB"]["DENSE_MODEL_NAME"])
EMB_DIM = encoder.get_sentence_embedding_dimension()


def compute_dense_vector(query_text: str) -> list:
    return encoder.encode(query_text).tolist()
