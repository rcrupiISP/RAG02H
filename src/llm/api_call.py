import json
import logging
import os
from typing import Any, Optional

import requests

from llm.prompt import get_prompt_1, get_prompt_2
from retrieval.search_qd import main_search
from retrieval.vdb_wrapper import SearchInVdb
from utility.read_config import get_config_from_path

dct_config = get_config_from_path("config.yaml")
LLM_MODEL_NAME = dct_config["RAG"]["LLM_MODEL_NAME"]


# Go to https://www.awanllm.com/, create an account and get the free secret key
# remember to run in the command line < export AWAN_API_KEY="your-api-key" >
# or edit in the run Python configuration as environment variable
def get_api_key(name: str = "AWAN_API_KEY") -> str:
    """
    Retrieves the API key from environment variables.

    Args:
        name (str): The name of the environment variable containing the API key.

    Returns:
        str: The API key.
    """
    return os.environ[name]


def basic_request(
    url: str,
    method: str,
    payload: dict[str, Any],
    headers: Optional[dict[str, Any]] = None,
) -> requests.Response:
    """
    Sends a basic HTTP request.

    Args:
        url (str): The URL for the request.
        method (str): The HTTP method (e.g., 'GET', 'POST').
        payload (dict[str, Any]): The data to send with the request.
        headers (Optional[dict[str, Any]]): Optional headers for the request.

    Returns:
        requests.Response: The response from the request.

    Raises:
        Exception: If the HTTP request results in an error.
    """
    _payload = json.dumps(payload)
    response = requests.request(
        method, url, headers=headers, data=_payload, verify=True
    )
    logging.debug(
        f"Raw response: \n{response.text}",
    )

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        logging.error(
            f"Status code: {response.status_code} when calling {url}.\n Response text: {response.text}"
        )
        raise Exception("HTTP Error") from e
    return response


def awan_model_completion(prompt: str) -> str:
    """Makes a completion request to the AWAN LLM API.

    Args:
        prompt (str): The text input for the LLM, expected to be a complete prompt.

    Returns:
        str: The response text from the LLM.
    """
    url = "https://api.awanllm.com/v1/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {get_api_key()}",
    }

    payload_dct = {
        "model": LLM_MODEL_NAME,
        "prompt": prompt,
        "max_tokens": 1024,
        "temperature": 0.7,
    }
    response = basic_request(
        url=url, method="POST", payload=payload_dct, headers=headers
    )
    response_str = json.loads(response.text)["choices"][0]["text"]
    return response_str


def awan_model_chat(usr_content_msg: str) -> str:
    """Makes a chat request to the AWAN LLM API.
    For more detail see https://www.awanllm.com/quick-start.

    Args:
        usr_content_msg (str): The user message content for the LLM.

    Returns:
        str: The response text from the LLM.
    """

    url = "https://api.awanllm.com/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {get_api_key()}",
    }

    payload_dct = {
        "model": LLM_MODEL_NAME,
        "max_tokens": 1024,
        "temperature": 0.7,
        "messages": [{"role": "user", "content": usr_content_msg}],
    }

    response = basic_request(
        url=url, method="POST", payload=payload_dct, headers=headers
    )
    response_str = json.loads(response.text)["choices"][0]["message"]["content"]

    return response_str


def main_api_call(searcher: SearchInVdb, question: str, rewriting: bool = True) -> str:
    """Handles the main API call flow including question refinement and searching.

    Args:
        searcher (SearchInVdb): The SearchInVdb instance used for searching.
        question (str): The user's question to process.
        rewriting (bool): Whether to refine the question using the LLM.

    Returns:
        str: The final response text from the LLM after processing.
    """
    logging.info(f'Starting RAG pipeline')
    if rewriting:
        p1 = get_prompt_1(question)
        logging.debug(f"question refinement prompt {p1}")
        _question = awan_model_completion(prompt=p1)
        logging.debug(f"Ameliorated question: {_question}")
    else:
        _question = question

    lst_points = main_search(searcher, query_text=_question)
    dct_points = {i: point.payload['text'] for i, point in enumerate(lst_points)}

    p2 = get_prompt_2(context=dct_points, question=_question)
    logging.debug(f"RAG prompt: {p2}")

    response_text = awan_model_chat(p2)
    logging.info(f'RAG pipeline ended')
    return response_text


if __name__ == "__main__":
    from dotenv import load_dotenv
    from qdrant_client.qdrant_client import QdrantClient

    load_dotenv()

    logging.basicConfig(level=logging.DEBUG)
    # logging.StreamHandler().setLevel(level=logging.INFO)

    logging.debug(
        f'Looking for vec db files in: {dct_config["VECTOR_DB"]["PATH_TO_FOLDER"]}'
    )
    client = QdrantClient(path=dct_config["VECTOR_DB"]["PATH_TO_FOLDER"])
    searcher = SearchInVdb(
        client=client, coll_name=dct_config["VECTOR_DB"]["COLLECTION_NAME"]
    )

    print(
        """Hi! Please provide here your question regarding one of the articles which have been loaded."""
    )
    question = input("your question >>>")
    # hi, my name is richmond jorge, i'm a software eng, well yaaa use to..ive been a scientist you know..sort of...been to NASA twice, yeah...great stuff.. ahahahhah...just wanna know whether there are any info a bout you know scintillators, I mean particle energy and stuff like that
    response = main_api_call(
        searcher, question, rewriting=dct_config["RAG"]["QUERY_REWRITING"]
    )
    print("RESPONSE: ", response)
