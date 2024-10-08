from typing import List

import markdownify
from bs4 import BeautifulSoup


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


def chunk_text(text: str, chunk_size: int = 300) -> List[str]:
    """
    ---- PLACEHOLDER VERSION ----
    ---- TO BE MODIFIED ----
    Chunks the input text into smaller segments of specified size.

    Args:
        text (str): The text to be chunked.
        chunk_size (int): The maximum number of words per chunk. Default is 300.

    Returns:
        List[str]: A list of text chunks.
    """
    words = text.split()
    chunks = [
        " ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)
    ]
    return chunks
