import requests
import feedparser
from bs4 import BeautifulSoup
import json
from time import sleep

# RSS feed URL
rss_url = "https://www.stik.at/en/realestate.xml"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

# Fetch and parse RSS feed
response = requests.get(rss_url, headers=headers)
feed = feedparser.parse(response.text)

print(f"Found {len(feed.entries)} entries in RSS feed.")

def get_listing_details(url):
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    data = {}
    
    # Page title (from <h1 class="page-header">)
    title_tag = soup.find('h1', class_='page-header')
    data['page_title'] = title_tag.get_text(strip=True) if title_tag else None
    
    # Address (from <h4 class="ia-item-view__sub-header">)
    address_tag = soup.find('h4', class_='ia-item-view__sub-header')
    data['address'] = address_tag.get_text(strip=True) if address_tag else None
    
    # Total price (from <div class="ia-item-view__price">)
    price_tag = soup.find('div', class_='ia-item-view__price')
    data['total_price'] = price_tag.get_text(strip=True) if price_tag else None
    
    # Category and Property Type (from first .ia-item-view__info__item)
    info_items = soup.find_all('span', class_='ia-item-view__info__item')
    if info_items:
        category_a = info_items[0].find('a')
        data['category'] = category_a.get_text(strip=True) if category_a else None
        type_b = info_items[0].find('b')
        data['property_type'] = type_b.get_text(strip=True) if type_b else None
    
    # Added date (from second .ia-item-view__info__item)
    if len(info_items) > 1:
        data['added_date'] = info_items[1].get_text(strip=True).replace('Added Date ', '') if info_items[1] else None
    
    # Descriptions (from tab panes)
    desc_de_tag = soup.find('div', id='tab-description')
    data['description_de'] = desc_de_tag.get_text(" ", strip=True) if desc_de_tag else None
    
    desc_en_tag = soup.find('div', id='tab-description-en')
    data['description_en'] = desc_en_tag.get_text(" ", strip=True) if desc_en_tag else None
    
    # Details and Prices (from all <table class="ia-item-view__table">)
    data['details'] = {}
    data['prices'] = {}
    tables = soup.find_all('table', class_='ia-item-view__table')
    current_section = 'details'  # Track section for prices vs details
    for table in tables:
        for tr in table.find_all('tr'):
            if 'Price Information' in tr.get_text():
                current_section = 'prices'
            elif 'Information' in tr.get_text():
                current_section = 'details'
            tds = tr.find_all('td')
            if len(tds) == 2:
                key = tds[0].get_text(strip=True).replace('{field_floor_space}', 'Total floor space')
                val = tds[1].get_text(strip=True)
                if 'Show phone number' in val or 'Send email' in val:
                    val = 'Hidden (loaded via JS)'
                if current_section == 'prices':
                    data['prices'][key.lower().replace(' ', '_')] = val
                else:
                    data['details'][key.lower().replace(' ', '_')] = val
    
    # Features (checked ones with fa-check-square)
    data['features'] = []
    feature_items = soup.find_all('div', class_='ia-item-view__features__item')
    for item in feature_items:
        if 'fa-check-square' in str(item):
            data['features'].append(item.get_text(strip=True))
    
    # Images (full res from <a href> in .fotorama, remove ~ if present)
    data['images'] = []
    gallery_items = soup.find_all('a', class_='ia-item-view__gallery__item')
    for item in gallery_items:
        img_href = item.get('href')
        if img_href:
            full_url = 'https:' + img_href if img_href.startswith('//') else img_href
            full_url = full_url.replace('~', '')  # Attempt to get full res
            data['images'].append(full_url)
    
    # Contact info (name visible, phone/email hidden)
    data['contact'] = {}
    if 'name' in data['details']:
        data['contact']['name'] = data['details']['name']
    data['contact']['phone'] = 'Hidden (use API: https://www.stik.at/packages/realestate/view.json?action=show-number&id=<ID>)'
    data['contact']['email'] = 'Hidden (button to send email)'
    
    # Fetch hidden phone/mobile via API if possible
    listing_id = url.split('/')[-1].split('.html')[0]
    api_url = "https://www.stik.at/packages/realestate/view.json"
    try:
        api_response = requests.get(api_url, params={'action': 'show-number', 'id': listing_id}, headers=headers)
        if api_response.status_code == 200:
            api_data = api_response.json()
            data['contact']['phone'] = api_data.get('phone', 'Not available')
            data['contact']['mobile'] = api_data.get('mobile', 'Not available')
    except Exception as e:
        print(f"API fetch failed for {url}: {e}")
    
    return data

listings_data = []

for entry in feed.entries:
    listing = {
        'rss_title': entry.title,
        'rss_description': entry.description,
        'rss_category': entry.category if 'category' in entry else None,
        'published': entry.published,
        'guid': entry.guid
    }
    details = get_listing_details(entry.link)
    listing.update(details)
    listings_data.append(listing)
    print(f"Scraped: {listing.get('page_title', 'Unknown')}")
    sleep(1)  # Polite delay to avoid rate limiting

# Save to JSON
with open('stik_listings.json', 'w', encoding='utf-8') as f:
    json.dump(listings_data, f, ensure_ascii=False, indent=4)

print("All listings have been saved to stik_listings.json")