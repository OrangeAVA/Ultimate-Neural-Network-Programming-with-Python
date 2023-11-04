import requests
from multiprocessing import Pool
from bs4 import BeautifulSoup


# Define the list of URLs to be scraped
urls = ["http://www.example.com/page1", "http://www.example.com/page2", ...]


def scrape(url):
    """
    Function to fetch a webpage and extract its title using BeautifulSoup
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')


    # Assume that the page title is contained within <title> tags
    title = soup.find('title').text
    return title


if __name__ == "__main__":
    # Define the multiprocessing pool
    with Pool(processes=4) as pool:
        # Use the pool's map method to apply the scrape function to every URL
        results = pool.map(scrape, urls)


    # Print the results
    for url, title in zip(urls, results):
        print(f"Title of {url} is {title}")