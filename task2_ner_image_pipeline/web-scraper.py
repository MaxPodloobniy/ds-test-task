"""
Pinterest image scraper for collecting animal images.

This script uses Playwright to scrape Pinterest search results
and download high-resolution images.
"""
import time
import os
import requests
import argparse
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

# Configuration
SCROLL_DISTANCE = 4000
INITIAL_WAIT = 5
SCROLL_WAIT = 2
DOWNLOAD_DELAY = 1
SCROLL_PROGRESS_INTERVAL = 10
DOWNLOAD_PROGRESS_INTERVAL = 100


def parse_args():
    """Parse command line arguments."""
    animals = [
        "chimpanzee", "coyote", "deer", "duck", "eagle", "elephant",
        "hedgehog", "hippopotamus", "kangaroo", "rhinoceros", "tiger"
    ]

    parser = argparse.ArgumentParser(description='Scrape Pinterest images')
    parser.add_argument('--animals', nargs='+', default=animals, help='List of animals to search')
    parser.add_argument('--scroll_times', type=int, default=50, help='Number of times to scroll')
    parser.add_argument('--output_dir', type=str, default='pinterest_dataset', help='Base directory to save images')

    return parser.parse_args()


def run_scraper(search_query, scroll_times):
    """Launches a browser, searches Pinterest, and scrapes unique post links."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        print("Opening Pinterest...")
        try:
            page.goto(f"https://www.pinterest.com/search/pins/?q={search_query}", timeout=30000)
        except Exception as e:
            print(f"Failed to load Pinterest for '{search_query}': {e}")
            return []
        time.sleep(INITIAL_WAIT)

        print("Scrolling the page...")
        post_links = set()

        # Collect links after each scroll
        for i in range(scroll_times):
            # Scroll down
            page.mouse.wheel(0, SCROLL_DISTANCE)
            time.sleep(SCROLL_WAIT)

            # Extract links after each scroll
            html = page.content()
            soup = BeautifulSoup(html, 'html.parser')

            for link in soup.find_all('a', href=True):
                if '/pin/' in link['href']:
                    post_links.add("https://www.pinterest.com" + link['href'])

            # Print progress
            if (i + 1) % SCROLL_PROGRESS_INTERVAL == 0:
                print(f"Scrolled {i + 1} times, found {len(post_links)} unique links")

        context.close()
        browser.close()

        return list(post_links)


def get_high_res_image(pin_url):
    """Extracts the highest resolution image URL from a Pinterest post."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(pin_url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Try to find the image in different ways
        meta_tag = soup.find("meta", property="og:image")
        if meta_tag:
            return meta_tag["content"]

        img_tag = soup.find("img", {"class": "hCL kVc L4E MIw"})
        if img_tag and "src" in img_tag.attrs:
            return img_tag["src"]

    except requests.exceptions.RequestException as e:
        print(f"Error loading page {pin_url}: {e}")
    except Exception as e:
        print(f"Unexpected error processing {pin_url}: {e}")
    return None


def download_images(post_links, save_path):
    """Downloads images from extracted Pinterest links and saves them locally."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    count = 0
    failed = 0

    print(f"Starting download of {len(post_links)} images...")

    for pin_url in post_links:
        img_url = get_high_res_image(pin_url)
        if img_url:
            try:
                response = requests.get(img_url, stream=True, timeout=10)
                response.raise_for_status()

                img_name = f"{count}.jpg"
                img_path = os.path.join(save_path, img_name)

                with open(img_path, "wb") as file:
                    for chunk in response.iter_content(1024):
                        file.write(chunk)

                count += 1
                if count % DOWNLOAD_PROGRESS_INTERVAL == 0:
                    print(f"Saved {count} images out of {len(post_links)}")
                time.sleep(DOWNLOAD_DELAY)
            except requests.exceptions.RequestException as e:
                failed += 1
                print(f"Error downloading {img_url}: {e}")
        else:
            failed += 1

    print(f"Download complete. Saved: {count}, Errors: {failed}")
    return count, failed


def main():
    args = parse_args()
    total_saved = 0
    total_failed = 0

    for animal in args.animals:
        print(f"\nSearching for images of {animal}")
        search_query = f"Photo of {animal}"

        save_path = os.path.join(args.output_dir, animal)

        post_links = run_scraper(search_query, args.scroll_times)
        print(f"Found {len(post_links)} unique links")

        saved, failed = download_images(post_links, save_path)
        total_saved += saved
        total_failed += failed

    print(f"\nOverall Statistics:")
    print(f"Total images saved: {total_saved}")
    print(f"Total errors: {total_failed}")


if __name__ == "__main__":
    main()