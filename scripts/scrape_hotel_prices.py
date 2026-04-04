"""
scrape_hotel_prices.py — Scrape hotel price tiers for Vietnam provinces.

Targets the CRITICAL data gap: 97.4% of OSM hotels have unknown price_tier.

Strategy:
  - Scrape Agoda hotel listings (public search results, no login)
  - Map hotel name → price tier: budget / mid_range / premium / luxury
  - Match against vn_hotels.csv by name similarity + province

Usage:
    python scripts/scrape_hotel_prices.py

Output:
    data/scraped/vn_hotel_prices.csv  — name, province, price_tier, price_usd, stars
    data/features/vn_hotels.csv       — updated with price data merged in

Requirements:
    pip install "scrapling[fetchers]" rapidfuzz
"""

import os
import sys
import time
import logging
import re
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE_DIR, "data", "scraped")
FEAT_DIR = os.path.join(BASE_DIR, "data", "features")
os.makedirs(OUT_DIR, exist_ok=True)

# Province → Agoda city ID mapping (public search params)
AGODA_TARGETS = [
    ("Ha Noi",    "https://www.agoda.com/city/hanoi-vn.html"),
    ("Ho Chi Minh", "https://www.agoda.com/city/ho-chi-minh-city-vn.html"),
    ("Da Nang",   "https://www.agoda.com/city/da-nang-vn.html"),
    ("Hue",       "https://www.agoda.com/city/hue-vn.html"),
    ("Khanh Hoa", "https://www.agoda.com/city/nha-trang-vn.html"),
    ("Lam Dong",  "https://www.agoda.com/city/da-lat-vn.html"),
    ("Quang Ninh","https://www.agoda.com/city/halong-vn.html"),
    ("Kien Giang","https://www.agoda.com/city/phu-quoc-vn.html"),
]

# Price thresholds in USD/night → VND tier
USD_TO_TIER = [
    (0,   30,  "budget",    30 * 24000),
    (30,  80,  "mid_range", 55 * 24000),
    (80,  200, "premium",  140 * 24000),
    (200, 9999,"luxury",   350 * 24000),
]

def classify_price(price_usd: float) -> tuple[str, int]:
    for lo, hi, tier, vnd in USD_TO_TIER:
        if lo <= price_usd < hi:
            return tier, vnd
    return "luxury", 350 * 24000


def parse_price_usd(text: str) -> float | None:
    """Extract numeric USD price from text like '$45', 'US$120', 'USD 80'."""
    m = re.search(r"[\$US]+\s*([\d,]+)", text.replace(",", ""))
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    m2 = re.search(r"([\d]+)\s*(?:USD|usd)", text)
    if m2:
        try:
            return float(m2.group(1))
        except ValueError:
            pass
    return None


def scrape_agoda_province(province: str, url: str) -> list[dict]:
    """Scrape hotel listing page for a Vietnam province from Agoda."""
    from scrapling.fetchers import StealthyFetcher

    records = []
    log.info(f"  Scraping {province}: {url}")
    try:
        # Agoda has anti-bot protection — use StealthyFetcher
        page = StealthyFetcher.fetch(
            url,
            headless=True,
            solve_cloudflare=False,
            timeout=45000,
            network_idle=True,
        )
        if page.status != 200:
            log.warning(f"    HTTP {page.status}")
            return records

        # Hotel cards — Agoda uses React, so check rendered DOM
        cards = page.css(
            "[data-selenium='hotel-item'], "
            "[class*='PropertyCard'], "
            "li[class*='hotel'], "
            "div[class*='listing-item']"
        )
        log.info(f"    Found {len(cards)} hotel cards")

        for card in cards:
            name = (card.css(
                "[data-selenium='hotel-name']::text, "
                "[class*='hotel-name']::text, "
                "h3::text, h2::text"
            ).get() or "").strip()
            if not name:
                continue

            # Price (nightly rate)
            price_raw = (card.css(
                "[class*='price']::text, [class*='rate']::text, "
                "[data-selenium='display-price']::text"
            ).get() or "").strip()

            # Stars
            star_raw = (card.css(
                "[class*='star']::attr(aria-label), "
                "[data-selenium='stars']::attr(title), "
                "[class*='StarRating']::text"
            ).get() or "").strip()

            stars = 0
            m = re.search(r"(\d)", star_raw)
            if m:
                stars = int(m.group(1))

            price_usd = parse_price_usd(price_raw)
            if price_usd is not None:
                tier, vnd = classify_price(price_usd)
            else:
                tier, vnd = "unknown", 0

            records.append({
                "name":               name,
                "province":           province,
                "stars":              stars,
                "price_usd_night":    price_usd,
                "price_tier":         tier,
                "estimated_price_vnd": vnd,
                "source":             "agoda.com",
            })

    except Exception as e:
        log.error(f"    Error: {e}")

    return records


def merge_with_osm_hotels(scraped: pd.DataFrame) -> None:
    """Merge scraped price data into the existing vn_hotels.csv by name similarity."""
    hotels_path = os.path.join(FEAT_DIR, "vn_hotels.csv")
    if not os.path.exists(hotels_path):
        log.warning(f"vn_hotels.csv not found at {hotels_path}, skipping merge")
        return

    try:
        from rapidfuzz import process, fuzz
    except ImportError:
        log.warning("rapidfuzz not installed — skipping name-match merge. Run: pip install rapidfuzz")
        return

    hotels = pd.read_csv(hotels_path)
    log.info(f"Merging scraped prices into {len(hotels)} OSM hotels ...")

    updated = 0
    for _, row in scraped.iterrows():
        if row["price_tier"] == "unknown":
            continue
        province_hotels = hotels[hotels["province"] == row["province"]]
        if province_hotels.empty:
            continue

        candidates = province_hotels["name"].tolist()
        match = process.extractOne(
            row["name"], candidates,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=70
        )
        if match:
            idx = province_hotels[province_hotels["name"] == match[0]].index[0]
            hotels.at[idx, "price_tier"] = row["price_tier"]
            hotels.at[idx, "estimated_price_vnd"] = row["estimated_price_vnd"]
            if row["stars"] > 0:
                hotels.at[idx, "star_rating"] = row["stars"]
            updated += 1

    hotels.to_csv(hotels_path, index=False)
    unknown_after = (hotels["price_tier"] == "unknown").sum()
    log.info(f"Merge complete: {updated} hotels updated, {unknown_after}/{len(hotels)} still unknown")


def main():
    all_records = []

    for province, url in AGODA_TARGETS:
        records = scrape_agoda_province(province, url)
        log.info(f"  {province}: {len(records)} hotels found")
        all_records.extend(records)
        time.sleep(2.0)  # polite delay between provinces

    if not all_records:
        log.error("No hotel data scraped. Check if Agoda is accessible / selectors need update.")
        return

    df = pd.DataFrame(all_records)
    out_path = os.path.join(OUT_DIR, "vn_hotel_prices.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    log.info(f"Saved → {out_path} ({len(df)} rows)")

    # Summary
    print(f"\nHotels scraped: {len(df)}")
    print(f"Price tier distribution:")
    print(df["price_tier"].value_counts().to_string())
    known = df[df["price_tier"] != "unknown"]
    print(f"Price coverage: {len(known)}/{len(df)} ({len(known)/len(df)*100:.1f}%)")

    # Merge into features
    merge_with_osm_hotels(df)


if __name__ == "__main__":
    main()
