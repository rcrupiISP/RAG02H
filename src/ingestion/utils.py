from typing import List, Dict
from logging import getLogger

import markdownify
from bs4 import BeautifulSoup

import tiktoken
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_text_splitters import HTMLHeaderTextSplitter, HTMLSectionSplitter

logger = getLogger("ingestion")


def convert_html_to_markdown(html_file: str) -> str:
    """Reads an HTML file and converts its content to Markdown format.

    Args:
        html_file (str): The path to the HTML file to be converted.

    Returns:
        str: The converted Markdown content.
    """
    with open(html_file, "r", encoding="utf-8") as file:
        html_content = file.read()

    # Parse the HTML using BeautifulSoup to clean it
    soup = BeautifulSoup(html_content, "html.parser")

    # Convert HTML to Markdown
    markdown_content = markdownify.markdownify(str(soup), heading_style="ATX")

    return markdown_content


def num_tokens_from_string(string: str, encoding_name: str = 'cl100k_base') -> int:
    """
    Function that returns the number of tokens in a text string.
    
    Args:
    string (str): The text for which we want to calculate the number of embeddings.
    encoding_name (str): The name of the encoding model. Default 'cl100k_base'.

    Returns:
        int: Number of tokens of the input string.
    """
    # https://stackoverflow.com/questions/76106366/how-to-use-tiktoken-in-offline-mode-computer
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def merge_chunks(docs: List[Document], threshold: int) -> List[Document]:
    """ 
    Merges consecutive chunks of documents based on matching metadata and a token threshold.

    Args:
        docs (List[Document]): A list of Document objects where each object represents a chunk of text. Each document is expected to have:
            - 'metadata': The header information, used to group paragraphs and subparagraphs.
            - 'page_content': The textual content of the document.
        threshold (int): The token threshold that determines when merging should occur. If a Document object has fewer tokens than this threshold and its metadata matches the previous or next document, it will be merged with the previous or next document.

    Returns:
        List[Document]: A list of Document objects where smaller chunks (below the threshold) with matching metadata have been merged with their neighboring documents.
    """
    merged_docs = []
    merged_chunks = [0] * len(docs)

    for i, doc in enumerate(docs):

        # If the current document has already been merged, skip it
        if merged_chunks[i] == 1:
            continue

        # Case 1: If the current document token count is below the threshold and it has matching metadata with the next one, merge the current document with the next one
        elif i < len(docs) - 1 and merged_chunks[i] == 0 and doc.metadata == docs[i+1].metadata and num_tokens_from_string(doc.page_content) < threshold:
            tmp_doc = Document(
                metadata = doc.metadata,
                page_content = doc.page_content + "\n" + docs[i+1].page_content)
            # Mark both documents as merged
            merged_chunks[i: i+2] = [1, 1]

        # Case 2: If the current document token count is below the threshold and it has matching metadata with the previous one, merge the current document with the previous one
        elif i > 0 and merged_chunks[i] == 0 and doc.metadata == docs[i-1].metadata and num_tokens_from_string(doc.page_content) < threshold:
            tmp_doc = Document(
                metadata = doc.metadata,
                page_content = docs[i-1].page_content + "\n" + doc.page_content)
            # Mark both documents as merged
            merged_chunks[i-1: i+1] = [1, 1]
            del merged_docs[-1] # Remove the last document because it will be replaced by the merged one

        else:
            tmp_doc = Document(
                metadata = doc.metadata,
                page_content = doc.page_content)
        
        # Append the document (merged or not) to the result list
        merged_docs.append(tmp_doc) 
    
    return merged_docs


def manage_subpar(docs: List[Document]) -> List[Dict]:
    """
    Processes a list of Document objects and organizes paragraphs and subparagraphs based on metadata.

    Args:
        docs (List[Document]): A list of Document objects where each object represents a chunk of text. Each document is expected to have:
            - 'metadata': The header information, used to group paragraphs and subparagraphs.
            - 'page_content': The textual content of the document.

    Returns:
        List[Dict]: A list of dictionaries where each entry contains:
            - 'text' (str): The content of the document with leading special characters (spaces, newlines, periods) removed.
            - 'par_ref' (int): The reference number of the paragraph.
            - 'subpar_ref' (int): The reference number of the subparagraph.
    """
    payloads = []
    headers = []
    par_ref = -1

    for doc in docs:
        if doc.metadata in headers:
            subpar_ref += 1 # Incrementing reference number of the subparagraph for each subsequent occurrence of the same header
        
        else:
            # TODO: The first Document has empty metadata because it has no headers. The case where other empty metadata not associated with the first in the list of documents are encountered is not handled.
            par_ref += 1    # Incrementing reference number of the paragraph for each new header encountered  
            subpar_ref = 0
            headers.append(doc.metadata)  

        payloads.append({
            "text": doc.page_content.lstrip(" \n."), # Removal of initial special characters
            "par_ref": par_ref,
            "subpar_ref": subpar_ref
        })

    return payloads

def chunk_text(text: str, chunking_mode: str = "recursive", max_chunk_size: int = 200, min_chunk_size: int = 50, chunk_overlap: int = 0, headers_level: int = 6, include_headers: bool = True) -> List[str]:  
    """
    Chunks the input text into smaller segments of specified size.

    Args:
        text (str): The text to be chunked.
        chunking_mode (str): The type of chunking to use. Options: 'recursive', 'markdown_specific', or 'html_specific'. Default is 'recursive'.
        max_chunk_size (int): The maximum number of token per chunk. Default is 200.
        min_chunk_size (int): The minimum number of token per chunk. Default is 50.
        chunk_overlap (int): The number of overlapping tokens between consecutive chunks. Default is 0.
        headers_level (int): Header level to use for splitting in 'markdown_specific' and 'html_specific' modes. Default is 6.
        include_headers (bool): If True, includes headers as part of chunk splitting in 'markdown_specific' and 'html_specific' modes. Default is True.

    Returns:
        List[str]: A list of text chunks.
    """
    splits = None

    if chunking_mode == "markdown_specific":
        # Markdown-based text splitting
        headers_to_split_on = [ ("#"*(i+1), f"Headers {i+1}") for i in range(headers_level)]
        
        if include_headers:
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
        else:
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
        
        try:
            splits = markdown_splitter.split_text(text)
        except ValueError as e:
            logger.error(f"Markdown-based text splitting is not possible, proceeding with no-format splitting")
        
    elif chunking_mode == "html_specific":
        # HTML-based text splitting
        headers_to_split_on = [ (f"h{i+1}", f"Headers {i+1}") for i in range(headers_level)]

        if include_headers:
            html_splitter = HTMLSectionSplitter(headers_to_split_on)
        else:
            html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)

        try:
            splits = html_splitter.split_text(text)
        except ValueError as e:
            logger.error(f"HTML-based text splitting is not possible, proceeding with no-format splitting")

    # Recursive text splitting
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        separators=["  \n", " \n", ".\n", ". ", ".", "\n", " ", ""],
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap
        )

    if splits:
        splits = text_splitter.split_documents(splits)
    else:
        splits = text_splitter.create_documents([text])

    # Preparation of chunks and removal of initial special characters
    splits = merge_chunks(splits, min_chunk_size)
    chunks = manage_subpar(splits)

    return chunks
