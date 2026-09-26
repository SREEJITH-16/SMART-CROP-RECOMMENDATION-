import os
import time
import requests
from urllib.parse import quote

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "crops")

os.makedirs(OUTPUT_DIR, exist_ok=True)

API = "https://commons.wikimedia.org/w/api.php"

CROPS = {
    "rice": "rice crop",
    "maize": "maize crop",
    "chickpea": "chickpea plant crop",
    "kidneybeans": "kidney bean plant",
    "pigeonpeas": "pigeon pea plant",
    "mothbeans": "moth bean plant",
    "mungbean": "mung bean plant",
    "blackgram": "black gram plant",
    "lentil": "lentil plant",
    "pomegranate": "pomegranate fruit tree",
    "banana": "banana plant",
    "mango": "mango tree fruit",
    "grapes": "grape vine grapes",
    "watermelon": "watermelon crop",
    "muskmelon": "muskmelon crop",
    "apple": "apple tree fruit",
    "orange": "orange tree fruit",
    "papaya": "papaya plant fruit",
    "coconut": "coconut palm fruit",
    "cotton": "cotton plant field",
    "jute": "jute plant crop",
    "coffee": "coffee plant crop",
    "groundnut": "groundnut peanut crop",
    "sesame": "sesame plant crop",
    "sugarcane": "sugarcane crop field",
    "tapioca": "cassava tapioca plant",
    "tomato": "tomato plant crop",
    "turmeric": "turmeric plant crop",
    "sorghum": "sorghum crop",
    "fingermillet": "finger millet crop",
    "pearlmillet": "pearl millet crop",
    "brinjal": "brinjal eggplant plant",
    "chilli": "chilli pepper plant crop",
    "okra": "okra plant crop",
    "greengram": "green gram mung bean plant",
}


HEADERS = {
    "User-Agent": "SmartCropRecommendation/1.0 educational-project"
}


def search_image(query):
    """Search Wikimedia Commons for an actual image."""

    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": 10,
        "prop": "imageinfo",
        "iiprop": "url|mime|size",
        "iiurlwidth": 900,
    }

    for attempt in range(4):
        try:
            response = requests.get(
                API,
                params=params,
                headers=HEADERS,
                timeout=30
            )

            if response.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue

            response.raise_for_status()

            data = response.json()

            pages = data.get("query", {}).get("pages", {})

            candidates = []

            for page in pages.values():

                title = page.get("title", "")

                imageinfo = page.get("imageinfo", [])

                if not imageinfo:
                    continue

                info = imageinfo[0]

                mime = info.get("mime", "")

                if not mime.startswith("image/"):
                    continue

                thumb = info.get("thumburl") or info.get("url")

                if not thumb:
                    continue

                candidates.append({
                    "title": title,
                    "url": thumb,
                    "mime": mime,
                    "width": info.get("thumbwidth", 0),
                    "height": info.get("thumbheight", 0)
                })

            if candidates:
                # Prefer reasonably large landscape/square images
                candidates.sort(
                    key=lambda x: (
                        x["width"] >= 500,
                        x["height"] >= 300,
                        x["width"] * x["height"]
                    ),
                    reverse=True
                )

                return candidates[0]

            return None

        except Exception as e:
            if attempt == 3:
                print("Search failed:", e)
                return None

            time.sleep(5)


def download_image(crop, query):
    output = os.path.join(OUTPUT_DIR, f"{crop}.jpg")

    if os.path.exists(output) and os.path.getsize(output) > 10_000:
        print(f"SKIP {crop}: already exists")
        return True

    result = search_image(query)

    if not result:
        print(f"ERR {crop}: no suitable Wikimedia image found")
        return False

    print(f"FOUND {crop}: {result['title']}")

    try:
        time.sleep(2)

        response = requests.get(
            result["url"],
            headers=HEADERS,
            timeout=60
        )

        if response.status_code == 429:
            print(f"ERR {crop}: rate limited while downloading")
            return False

        response.raise_for_status()

        with open(output, "wb") as f:
            f.write(response.content)

        size_kb = os.path.getsize(output) // 1024

        print(f"OK   {crop}: {crop}.jpg ({size_kb} KB)")

        return True

    except Exception as e:
        print(f"ERR {crop}: {e}")
        return False


def main():

    print("Searching Wikimedia Commons for actual crop photographs...")
    print()

    success = 0
    failed = 0

    for crop, query in CROPS.items():

        if download_image(crop, query):
            success += 1
        else:
            failed += 1

        # Be respectful to Wikimedia
        time.sleep(3)

    print()
    print("=" * 60)
    print(f"Finished: {success} downloaded/present, {failed} failed")
    print(f"Images are in: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()