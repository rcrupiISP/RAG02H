import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from utils.read_config import get_config_from_path

dct_config = get_config_from_path("config.yaml")

tokenizer = AutoTokenizer.from_pretrained(
    dct_config["PRE_TRAINED_EMB"]["SPARSE_MODEL_NAME"]
)
model = AutoModelForMaskedLM.from_pretrained(
    dct_config["PRE_TRAINED_EMB"]["SPARSE_MODEL_NAME"]
)


# TODO: this implementation is just a placeholder, to be modified!
def __compute_vector(text):
    """
    Computes a vector from logits and attention mask using ReLU, log, and max operations.
    Taken from Qdrant documentation: https://qdrant.tech/articles/sparse-vectors/
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
    out = {"indices": q_vec.nonzero().numpy().flatten().tolist()}
    out["values"] = q_vec.detach().numpy()[out["indices"]].tolist()
    return out
