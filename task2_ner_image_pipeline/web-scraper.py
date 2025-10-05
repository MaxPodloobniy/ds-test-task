from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import requests
import time
import os


def create_folder(path):
    """Creates a folder if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def run_scraper(search_query, scroll_times):
    """Launches a browser, searches Pinterest, and scrapes unique post links."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        print("Opening Pinterest...")
        page.goto(f"https://www.pinterest.com/search/pins/?q={search_query}")
        time.sleep(5)

        print("Scrolling the page...")
        post_links = set()

        # Collect links after each scroll
        for i in range(scroll_times):
            # Scroll down
            page.mouse.wheel(0, 4000)
            time.sleep(2)

            # Extract links after each scroll
            html = page.content()
            soup = BeautifulSoup(html, 'html.parser')

            for link in soup.find_all('a', href=True):
                if '/pin/' in link['href']:
                    post_links.add("https://www.pinterest.com" + link['href'])

            # Print progress every 10 scrolls
            if (i + 1) % 10 == 0:
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
    create_folder(save_path)
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
                if count % 100 == 0:
                    print(f"Saved {count} images out of {len(post_links)}")
                time.sleep(1)
            except requests.exceptions.RequestException as e:
                failed += 1
                print(f"Error downloading {img_url}: {e}")
        else:
            failed += 1

    print(f"Download complete. Saved: {count}, Errors: {failed}")
    return count, failed


if __name__ == "__main__":
    animals = [
        "antelope", "bison", "chimpanzee", "cow", "coyote", "deer", "donkey", "duck",
        "eagle", "elephant", "gorilla", "hedgehog", "hippopotamus", "hyena",
        "kangaroo", "racoon", "rhinoceros", "tiger", "turkey", "wolf"
    ]
    scroll_times = 50
    base_path = "pinterest_dataset"

    total_saved = 0
    total_failed = 0

    for animal in animals:
        print(f"\nSearching for images of {animal}")
        search_query = f"Photo of {animal}"

        save_path = os.path.join(base_path, animal)

        post_links = run_scraper(search_query, scroll_times)
        print(f"Found {len(post_links)} unique links")

        saved, failed = download_images(post_links, save_path)
        total_saved += saved
        total_failed += failed

    print(f"\nOverall Statistics:")
    print(f"Total images saved: {total_saved}")
    print(f"Total errors: {total_failed}")
