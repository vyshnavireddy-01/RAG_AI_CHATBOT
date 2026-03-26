import requests
import xml.etree.ElementTree as ET
import ftfy
import time

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


# =========================
# Setup Selenium driver once
# =========================
def create_driver():
    options = Options()
    options.add_argument("--headless")           # run in background, no window
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--log-level=3")        # suppress chrome logs
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    service = Service(ChromeDriverManager().install())
    driver  = webdriver.Chrome(service=service, options=options)
    return driver


# =========================
# Get all URLs from sitemap
# =========================
def get_sitemap_urls(sitemap_url):
    urls = []

    try:
        response = requests.get(sitemap_url, timeout=10)
        response.encoding = "utf-8"
        root      = ET.fromstring(response.content)
        namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}

        for url in root.findall(".//ns:loc", namespace):
            link = url.text.strip()
            if any(x in link for x in ["tag", "author", "feed", "wp-json"]):
                continue
            urls.append(link)

    except Exception as e:
        print(f"Sitemap error: {e}")

    return urls


# =========================
# Clean text
# =========================
def clean_text(text):
    if not text:
        return ""
    text = ftfy.fix_text(text)
    text = " ".join(text.split())
    return text.strip()


# =========================
# Extract text using Selenium
# Reads JS-rendered content: testimonials, team, office hours, services
# =========================
def extract_text(driver, url):
    try:
        driver.get(url)

        # Wait for page body to fully load
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        # Extra wait for JS to render dynamic content
        time.sleep(3)

        content_parts = []

        # Page title
        title = clean_text(driver.title)
        if title:
            content_parts.append(title)

        # All headings
        for tag in ["h1", "h2", "h3", "h4"]:
            elements = driver.find_elements(By.TAG_NAME, tag)
            for el in elements:
                text = clean_text(el.text)
                if text and len(text) > 2:
                    content_parts.append(text)

        # Paragraphs
        for el in driver.find_elements(By.TAG_NAME, "p"):
            text = clean_text(el.text)
            if len(text) > 40:
                content_parts.append(text)

        # List items — captures service bullet points
        for el in driver.find_elements(By.TAG_NAME, "li"):
            text = clean_text(el.text)
            if len(text) > 30:
                content_parts.append(text)

        # Blockquotes — captures testimonials
        for el in driver.find_elements(By.TAG_NAME, "blockquote"):
            text = clean_text(el.text)
            if len(text) > 20:
                content_parts.append(text)

        # Divs with key classes — captures JS-rendered sections
        # (testimonials, team cards, office info, pricing)
        for el in driver.find_elements(By.CSS_SELECTOR,
            "div[class*='testimonial'], div[class*='review'], "
            "div[class*='team'], div[class*='member'], "
            "div[class*='office'], div[class*='hour'], "
            "div[class*='service'], div[class*='card'], "
            "div[class*='about'], div[class*='mission'], "
            "div[class*='vision'], div[class*='story'], "
            "section, article"
        ):
            text = clean_text(el.text)
            if len(text) > 50:
                content_parts.append(text)

        # Spans — captures labels, tags, small info
        for el in driver.find_elements(By.TAG_NAME, "span"):
            text = clean_text(el.text)
            if len(text) > 40:
                content_parts.append(text)

        # Remove duplicates while preserving order
        seen = set()
        unique_parts = []
        for part in content_parts:
            if part not in seen:
                seen.add(part)
                unique_parts.append(part)

        full_text = "\n".join(unique_parts)
        return full_text

    except Exception as e:
        print(f"Failed to crawl {url}: {e}")
        return ""


# =========================
# Main crawl function
# =========================
def crawl_website(sitemap_url):
    print("Reading sitemap...")
    urls = get_sitemap_urls(sitemap_url)
    print(f"Found {len(urls)} pages")

    driver = create_driver()
    pages  = []

    try:
        for url in urls:
            print("Crawling:", url)
            page_text = extract_text(driver, url)

            if page_text:
                pages.append({
                    "url":  url,
                    "text": page_text,
                    "type": "blog" if "/blog" in url.lower() else "page"
                })

    finally:
        driver.quit()

    print(f"Successfully crawled {len(pages)} pages")
    return pages