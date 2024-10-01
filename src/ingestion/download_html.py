import requests
import os
import arxiv

from logging import getLogger

logger = getLogger("ingestion")


# Function to list paper links from arXiv based on a keyword
def list_arxiv_links(keyword, max_results=10):

    # Create a client for searching arXiv
    client = arxiv.Client()

    # Search for papers using the arxiv library
    search = arxiv.Search(
        query=keyword, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
    )
    # List the paper links
    paper_links = []
    logger.info("Query results:")
    for i, result in enumerate(client.results(search)):
        paper_links.append(result.entry_id)
        logger.info(f"{i}. Title: {result.title[:20]}, Link: {result.entry_id}")

    return paper_links


# Function to download HTML from a website and save it to a folder
def download_html_from_url(url, save_dir, filename="downloaded_page.html"):
    # Fetch the webpage
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Create the save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Saving documents in directory: {save_dir}")

        # Save the HTML content to the specified folder
        file_path = os.path.join(save_dir, filename)

        if os.path.exists(file_path):
            logger.info(f"File {filename} already exists.")
        else:
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(response.text)

            logger.info(f"Downloaded and saved: {filename}")
    else:
        logger.error(
            f"Failed to download the webpage. Status code: {response.status_code}"
        )


def main_html_download(keyword: str, output_dir: str):
    # Call the function and list paper links
    arxiv_links = list_arxiv_links(keyword, max_results=5)

    # URL of the website to download
    for url in arxiv_links:
        url_html = url.replace("//arxiv.org", "//ar5iv.org")
        # Call the function to download the HTML
        download_html_from_url(
            url_html, output_dir, filename=url.split("/")[-1] + ".html"
        )


if __name__ == "__main__":
    from utils.read_config import get_config_from_path

    logger.setLevel("INFO")
    keyword = "riccardo crupi"

    # Set the folder to save the downloaded HTML
    dct_config = get_config_from_path("config.yaml")
    project_save_dir = dct_config["INPUT_DATA"]["PATH_TO_FOLDER"]

    main_html_download(keyword=keyword, output_dir=project_save_dir)
