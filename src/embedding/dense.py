# todo: move to config
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
TOKENIZER_MODEL_NAME = "naver/splade-cocondenser-ensembledistil"

from sentence_transformers import SentenceTransformer

# MODEL TO GENERATE DENSE VECTORS
encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)  # Model to create embeddings
EMB_DIM = encoder.get_sentence_embedding_dimension()


def compute_dense_vector(query_text: str) -> list:
    return encoder.encode(query_text).tolist()
