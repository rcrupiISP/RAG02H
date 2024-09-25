# todo: move to config
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
TOKENIZER_MODEL_NAME = "naver/splade-cocondenser-ensembledistil"

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# MODEL TO GENERATE SPARSE VECTORS
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(TOKENIZER_MODEL_NAME)


def __compute_vector(text):
    """
    Computes a vector from logits and attention mask using ReLU, log, and max operations.
    from https://qdrant.tech/articles/sparse-vectors/
    """
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    output = model(**tokens)
    logits, attention_mask = output.logits, tokens.attention_mask
    relu_log = torch.log(1 + torch.relu(logits))
    weighted_log = relu_log * attention_mask.unsqueeze(-1)
    max_val, _ = torch.max(weighted_log, dim=1)
    vec = max_val.squeeze()

    return vec, tokens


def compute_sparse_vector(query_text: str) -> dict[str, list]:
    q_vec, q_tokens = __compute_vector(query_text)
    out = {'indices': q_vec.nonzero().numpy().flatten().tolist()}
    out['values'] = q_vec.detach().numpy()[out['indices']].tolist()
    return out
