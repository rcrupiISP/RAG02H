import requests
import os
import arxiv


# Function to list paper links from arXiv based on a keyword
def list_arxiv_links(keyword, max_results=10):

    # Create a client for searching arXiv
    client = arxiv.Client()

    # Search for papers using the arxiv library
    search = arxiv.Search(
        query=keyword,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    # List the paper links
    paper_links = []
    for result in client.results(search):
        paper_links.append(result.entry_id)
        print(f"Title: {result.title}")
        print(f"Link: {result.entry_id}\n")

    return paper_links


# Function to download HTML from a website and save it to a folder
def download_html_from_url(url, save_dir, filename="downloaded_page.html"):
    # Fetch the webpage
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Create the save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Save the HTML content to the specified folder
        file_path = os.path.join(save_dir, filename)

        if os.path.exists(file_path):
            print("this file path already exists: ", file_path)
        else:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(response.text)

            print(f"Downloaded HTML and saved to {file_path}")
    else:
        print(f"Failed to download the webpage. Status code: {response.status_code}")


if __name__ == "__main__":
    keyword = 'riccardo crupi'

    # Call the function and list paper links
    arxiv_links = list_arxiv_links(keyword, max_results=5)

    # URL of the website to download
    for url in arxiv_links:
        url_html = url.replace('//arxiv.org', '//ar5iv.org')
        # Set the folder to save the downloaded HTML (e.g., in your project's 'data/webpages' folder)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        project_save_dir = os.path.join(project_root, 'data', 'docs')

        # Call the function to download the HTML
        download_html_from_url(url_html, project_save_dir, filename=url.split('/')[-1] + '.html')
