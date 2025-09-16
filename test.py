import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import json

# === CONFIG ===
rss_feed_url = "https://www.stik.at/en/realestate.xml"  # RSS XML feed

# === HELPER FUNCTION TO SCRAPE ONE LISTING ===
def scrape_listing(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    listing = {
        "url": url,
        "title": None,
        "description": None,
        "prices": {},
        "details": {},
        "features": [],
        "images": [],
        "contact": {}
    }

    # Title
    title_tag = soup.find("h1", class_="page-header")
    if title_tag:
        listing["title"] = title_tag.text.strip()

    # Description (English)
    desc_tag = soup.find("div", id="tab-description-en")
    if desc_tag:
        listing["description"] = desc_tag.get_text(separator="\n").strip()

    # Price and other details
    tables = soup.find_all("table", class_="ia-item-view__table")
    for table in tables:
        for row in table.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) == 2:
                key = cols[0].text.strip()
                value = cols[1].text.strip()
                key_lower = key.lower()
                if key_lower in ["rent", "operating costs", "rental costs (all)", "deposit"]:
                    listing["prices"][key] = value
                elif key_lower in ["name", "phone", "mobile", "email"]:
                    listing["contact"][key] = value
                else:
                    listing["details"][key] = value

    # Features
    feature_divs = soup.find_all("div", class_="ia-item-view__features__item")
    for f in feature_divs:
        listing["features"].append(f.get_text(strip=True))

    # Images
    gallery_items = soup.select(".ia-item-view__gallery__item img")
    for img in gallery_items:
        img_url = img.get("src")
        if img_url:
            listing["images"].append(img_url)

    return listing

# === MAIN SCRIPT ===
def main():
    # Load RSS feed XML
    rss_response = requests.get(rss_feed_url)
    root = ET.fromstring(rss_response.content)

    # Extract listing URLs from RSS
    listing_urls = []
    for item in root.findall(".//item"):
        link_tag = item.find("link")
        if link_tag is not None and link_tag.text:
            listing_urls.append(link_tag.text.strip())

    # Scrape each listing
    all_listings = []
    for url in listing_urls:
        print(f"Scraping: {url}")
        try:
            listing_data = scrape_listing(url)
            all_listings.append(listing_data)
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")

    # Save to JSON
    with open("stik_listings.json", "w", encoding="utf-8") as f:
        json.dump(all_listings, f, ensure_ascii=False, indent=2)

    print(f"Scraped {len(all_listings)} listings. Saved to stik_listings.json")

if __name__ == "__main__":
    main()
