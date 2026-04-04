"""
scrape_vn_places.py — Scrape Vietnam tourist place info from vietnam.travel (official).

Targets data gaps identified in EDA:
  - Entry fees for known places (OSM has 75.5% free/unknown)
  - Opening hours for places (current data uses defaults 7-17)
  - Short descriptions for tourist places
  - Missing provinces (Hue, Hoi An, Can Tho under-represented)

Usage:
    python scripts/scrape_vn_places.py

Output:
    data/scraped/vn_places_detail.csv
    data/scraped/vn_places_extra.csv  (newly discovered places not in base list)

Requirements:
    pip install "scrapling[fetchers]"
"""

import os
import sys
import time
import json
import logging
import pandas as pd
from urllib.parse import urljoin

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE_DIR, "data", "scraped")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Target sources (all public, no login required)
# ---------------------------------------------------------------------------
# 1. vietnam.travel — official Vietnam Tourism website
# 2. visitmekong.com — Mekong region travel
# 3. Vietnam province tourism portals (static pages)

PROVINCE_QUERY_MAP = {
    "Ha Noi":     "hanoi",
    "Ho Chi Minh":"ho-chi-minh-city",
    "Da Nang":    "da-nang",
    "Hue":        "hue",
    "Hoi An":     "hoi-an",
    "Ha Long":    "ha-long",
    "Nha Trang":  "nha-trang",
    "Da Lat":     "da-lat",
    "Phu Quoc":   "phu-quoc",
    "Ninh Binh":  "ninh-binh",
    "Sapa":       "sapa",
    "Can Tho":    "can-tho",
    "Hoi An":     "hoi-an",
    "Mui Ne":     "mui-ne",
    "Ha Giang":   "ha-giang",
}

VIETNAM_TRAVEL_BASE = "https://vietnam.travel"
SEARCH_URL = "https://vietnam.travel/things-to-do/attractions"


def scrape_vietnam_travel() -> list[dict]:
    """Scrape attraction listings from vietnam.travel."""
    from scrapling.fetchers import Fetcher

    records = []
    page_num = 1
    max_pages = 10  # cap to avoid overloading

    log.info("Scraping vietnam.travel/things-to-do/attractions ...")

    while page_num <= max_pages:
        url = f"{SEARCH_URL}?page={page_num}"
        try:
            page = Fetcher.get(url, impersonate="chrome", timeout=30)
            if page.status != 200:
                log.warning(f"  Page {page_num}: HTTP {page.status} — stopping")
                break

            # Extract attraction cards
            cards = page.css("article.views-row, .attraction-card, .card, article")
            if not cards:
                log.info(f"  Page {page_num}: no cards found — stopping")
                break

            new_on_page = 0
            for card in cards:
                try:
                    name = (card.css("h2::text, h3::text, .title::text").get() or "").strip()
                    if not name:
                        continue

                    link_el = card.css("a::attr(href)").get() or ""
                    link = urljoin(VIETNAM_TRAVEL_BASE, link_el) if link_el else ""

                    desc = (card.css(
                        "p::text, .description::text, .field-body::text, .summary::text"
                    ).get() or "").strip()

                    location = (card.css(
                        ".location::text, .province::text, [class*='location']::text"
                    ).get() or "").strip()

                    category = (card.css(
                        ".category::text, .type::text, [class*='type']::text"
                    ).get() or "").strip()

                    records.append({
                        "name":        name,
                        "description": desc[:500] if desc else "",
                        "province":    location,
                        "category":    category,
                        "source_url":  link,
                        "source":      "vietnam.travel",
                    })
                    new_on_page += 1
                except Exception as e:
                    log.debug(f"  Card parse error: {e}")

            log.info(f"  Page {page_num}: +{new_on_page} attractions (total: {len(records)})")

            # Check for next page
            next_link = page.css("a[rel='next']::attr(href), .pager-next a::attr(href)").get()
            if not next_link:
                log.info(f"  No next page — done")
                break

            page_num += 1
            time.sleep(1.5)  # polite delay

        except Exception as e:
            log.error(f"  Page {page_num} error: {e}")
            break

    return records


def scrape_place_details(url: str) -> dict:
    """Scrape detail page for a single attraction: opening hours, entry fee, coordinates."""
    from scrapling.fetchers import Fetcher

    detail = {"opening_hours_raw": "", "entry_fee_raw": "", "address": "", "lat": None, "lon": None}
    try:
        page = Fetcher.get(url, impersonate="chrome", timeout=20)
        if page.status != 200:
            return detail

        # Opening hours
        hours_text = page.css(
            "[class*='hour']::text, [class*='opening']::text, "
            ".field-opening-hours::text, dt:contains('Opening') + dd::text"
        ).get() or ""
        detail["opening_hours_raw"] = hours_text.strip()

        # Entry fee
        fee_text = page.css(
            "[class*='fee']::text, [class*='admission']::text, [class*='price']::text, "
            ".field-entry-fee::text, dt:contains('Entrance') + dd::text"
        ).get() or ""
        detail["entry_fee_raw"] = fee_text.strip()

        # Address
        addr = page.css(
            "[class*='address']::text, .location::text, .field-address::text"
        ).get() or ""
        detail["address"] = addr.strip()

        # Coordinates from JSON-LD
        json_lds = page.css("script[type='application/ld+json']::text").getall()
        for jld in json_lds:
            try:
                data = json.loads(jld)
                geo = data.get("geo") or (data.get("@graph") or [{}])[0].get("geo", {})
                if geo:
                    detail["lat"] = float(geo.get("latitude", 0)) or None
                    detail["lon"] = float(geo.get("longitude", 0)) or None
                    break
            except Exception:
                pass

    except Exception as e:
        log.debug(f"  Detail scrape error for {url}: {e}")

    return detail


def scrape_foody_categories() -> list[dict]:
    """
    Scrape restaurant categories from foody.vn to fill the 86.9% cuisine gap.
    Targets the public restaurant listing pages (no login required).
    """
    from scrapling.fetchers import Fetcher

    records = []
    cities = [
        ("Ha Noi",    "ha-noi"),
        ("Ho Chi Minh", "ho-chi-minh"),
        ("Da Nang",   "da-nang"),
        ("Hue",       "hue"),
        ("Khanh Hoa", "nha-trang"),
    ]

    log.info("Scraping foody.vn for restaurant cuisine data ...")

    for province, city_slug in cities:
        url = f"https://www.foody.vn/{city_slug}/mon-an"
        try:
            page = Fetcher.get(url, impersonate="chrome", timeout=25)
            if page.status != 200:
                log.warning(f"  {province}: HTTP {page.status}")
                continue

            # Restaurant cards
            cards = page.css(".res-item, .restaurant-item, [class*='item']")
            for card in cards:
                name = (card.css("[class*='name']::text, h3::text, h2::text").get() or "").strip()
                if not name or len(name) < 3:
                    continue

                cuisine = (card.css("[class*='category']::text, [class*='cuisine']::text, "
                                    "[class*='type']::text, small::text").get() or "").strip()

                address = (card.css("[class*='address']::text, [class*='location']::text").get() or "").strip()

                records.append({
                    "name":     name,
                    "cuisine":  cuisine[:100] if cuisine else "unknown",
                    "province": province,
                    "address":  address[:200] if address else "",
                    "source":   "foody.vn",
                })

            log.info(f"  {province}: {len([r for r in records if r['province']==province])} restaurants")
            time.sleep(1.0)

        except Exception as e:
            log.error(f"  {province} error: {e}")

    return records


def parse_entry_fee_vnd(raw: str) -> int:
    """Convert raw entry fee text to VND integer. Returns 0 if free/unknown."""
    if not raw:
        return 0
    raw_lower = raw.lower()
    if any(w in raw_lower for w in ["free", "no charge", "miễn phí", "0"]):
        return 0

    import re
    # Look for numbers followed by currency indicators
    # Patterns: "50,000 VND", "50.000đ", "50000 dong", "USD 5"
    vnd_match = re.search(r"([\d,\.]+)\s*(?:vnd|đ|dong|₫)", raw_lower)
    if vnd_match:
        num_str = vnd_match.group(1).replace(",", "").replace(".", "")
        try:
            return int(num_str)
        except ValueError:
            pass

    usd_match = re.search(r"(?:usd|\$)\s*([\d,\.]+)", raw_lower)
    if usd_match:
        try:
            usd = float(usd_match.group(1).replace(",", ""))
            return int(usd * 24000)  # approximate VND conversion
        except ValueError:
            pass

    return 0


def main():
    all_places = []
    restaurant_data = []

    # ── Step 1: Scrape vietnam.travel listings ────────────────────────────
    vt_places = scrape_vietnam_travel()
    log.info(f"vietnam.travel: {len(vt_places)} attractions found")
    all_places.extend(vt_places)

    # ── Step 2: Enrich top places with detail pages ───────────────────────
    detail_limit = 30  # scrape details for first N to avoid long runtime
    enriched = 0
    for place in all_places[:detail_limit]:
        if place.get("source_url"):
            detail = scrape_place_details(place["source_url"])
            place.update(detail)
            place["entry_fee_vnd"] = parse_entry_fee_vnd(detail.get("entry_fee_raw", ""))
            enriched += 1
            time.sleep(0.8)

    log.info(f"Detail enrichment: {enriched} places enriched")

    # ── Step 3: Scrape foody.vn for cuisine classification ────────────────
    restaurant_data = scrape_foody_categories()
    log.info(f"foody.vn: {len(restaurant_data)} restaurant entries found")

    # ── Save outputs ──────────────────────────────────────────────────────
    if all_places:
        df_places = pd.DataFrame(all_places)
        out_path = os.path.join(OUT_DIR, "vn_places_scraped.csv")
        df_places.to_csv(out_path, index=False, encoding="utf-8")
        log.info(f"Saved → {out_path} ({len(df_places)} rows)")

        # Summary
        print(f"\nPlaces scraped: {len(df_places)}")
        if "province" in df_places.columns:
            print(df_places["province"].value_counts().head(10).to_string())
    else:
        log.warning("No places scraped — check network / selectors")

    if restaurant_data:
        df_rest = pd.DataFrame(restaurant_data)
        out_path = os.path.join(OUT_DIR, "vn_restaurants_scraped.csv")
        df_rest.to_csv(out_path, index=False, encoding="utf-8")
        log.info(f"Saved → {out_path} ({len(df_rest)} rows)")

        print(f"\nRestaurants scraped: {len(df_rest)}")
        if "cuisine" in df_rest.columns:
            known = df_rest[df_rest["cuisine"] != "unknown"]
            print(f"  cuisine known: {len(known)} / {len(df_rest)} ({len(known)/len(df_rest)*100:.1f}%)")


if __name__ == "__main__":
    main()
