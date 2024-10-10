from typing import List
from logging import getLogger

import markdownify
from bs4 import BeautifulSoup

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_text_splitters import HTMLSectionSplitter
from langchain_text_splitters import HTMLHeaderTextSplitter

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


def chunk_text(text: str, chunking_mode: str = "recursive", chunk_size: int = 200, chunk_overlap: int = 0, headers_level: int = 6, include_headers: bool = True) -> List[str]:  
    """
    Chunks the input text into smaller segments of specified size.

    Args:
        text (str): The text to be chunked.
        chunking_mode (str): The type of chunking to use. Options: 'recursive', 'markdown_specific', or 'html_specific'. Default is 'recursive'.
        chunk_size (int): The maximum number of token per chunk. Default is 200.
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
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
        )

    if not splits:
        splits = text_splitter.create_documents([text])

    splits = text_splitter.split_documents(splits)

    # Preparation of chunks and removal of initial special characters
    chunks = [doc.page_content.lstrip(" \n.") for doc in splits]

    return chunks
