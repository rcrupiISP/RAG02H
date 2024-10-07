import markdownify
from bs4 import BeautifulSoup


# Function to read the HTML file and convert it to markdown using markdown-it-py
def convert_html_to_markdown(html_file):
    with open(html_file, "r", encoding="utf-8") as file:
        html_content = file.read()

    # Parse the HTML using BeautifulSoup to clean it
    soup = BeautifulSoup(html_content, "html.parser")

    # Convert HTML to Markdown
    markdown_content = markdownify.markdownify(str(soup), heading_style="ATX")

    return markdown_content


# Function to chunk the markdown content
def chunk_text(text, chunk_size=300):
    # placeholder version
    words = text.split()
    chunks = [
        " ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)
    ]
    return chunks
