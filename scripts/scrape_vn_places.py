"""
scrape_vn_places.py — Scrape Vietnam tourist place info from vietnam.travel (official).

Targets data gaps:
  - Descriptions for tourist places (vn_tourist_places.csv has none)
  - Additional provinces not well-covered (Hue, Hoi An, Can Tho, etc.)
  - Entry fee signals and opening hours from official source

Strategy:
  1. Scrape vietnam.travel city pages (/places-to-go/...) for place descriptions
  2. Scrape vietnam.travel things-to-do sub-pages (festivals, cuisine, etc.)
  3. Save as supplementary data; merge description into vn_tourist_places.csv

Usage:
    python scripts/scrape_vn_places.py

Output:
    data/scraped/vn_places_detail.csv     — place name + description + province
    data/features/vn_tourist_places.csv   — updated with description column

Requirements:
    pip install "scrapling[fetchers]"
"""

import os, re, time, logging
import pandas as pd
from urllib.parse import urljoin

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE_DIR, "data", "scraped")
FEAT_DIR = os.path.join(BASE_DIR, "data", "features")
os.makedirs(OUT_DIR, exist_ok=True)

VT_BASE = "https://vietnam.travel"

# vietnam.travel city pages — confirmed accessible (200)
CITY_PAGES = [
    ("Ha Noi",      "/places-to-go/northern-vietnam/ha-noi"),
    ("Ha Giang",    "/places-to-go/northern-vietnam/ha-giang"),
    ("Ha Long",     "/places-to-go/northern-vietnam/ha-long"),
    ("Ninh Binh",   "/places-to-go/northern-vietnam/ninh-binh"),
    ("Sapa",        "/places-to-go/northern-vietnam/sapa"),
    ("Da Nang",     "/places-to-go/central-vietnam/da-nang"),
    ("Hue",         "/places-to-go/central-vietnam/hue"),
    ("Hoi An",      "/places-to-go/central-vietnam/hoi-an"),
    ("Da Lat",      "/places-to-go/central-vietnam/dalat"),
    ("Nha Trang",   "/places-to-go/central-vietnam/nha-trang"),
    ("Phong Nha",   "/places-to-go/central-vietnam/phong-nha"),
    ("Ho Chi Minh", "/places-to-go/southern-vietnam/ho-chi-minh-city"),
    ("Phu Quoc",    "/places-to-go/southern-vietnam/phu-quoc"),
    ("Con Dao",     "/places-to-go/southern-vietnam/con-dao"),
    ("Mekong",      "/places-to-go/southern-vietnam/mekong-delta"),
]

# Sub-pages within city pages that have linked place articles
THINGS_TO_DO_PAGES = [
    "https://vietnam.travel/things-to-do",
    "https://vietnam.travel/things-to-do/cuisine",
    "https://vietnam.travel/things-to-do/nature",
    "https://vietnam.travel/things-to-do/culture",
    "https://vietnam.travel/things-to-do/beaches",
    "https://vietnam.travel/things-to-do/adventure",
]


def scrape_city_page(province: str, path: str) -> list[dict]:
    """Scrape a vietnam.travel city page for place names and descriptions."""
    from scrapling.fetchers import Fetcher

    records = []
    url = VT_BASE + path
    try:
        page = Fetcher.get(url, impersonate="chrome", timeout=20)
        if page.status != 200:
            log.warning(f"  {province} ({url}): HTTP {page.status}")
            return records

        # Extract all article links within this page
        article_links = set()
        for a in page.css("a[href]"):
            href = a.attrib.get("href", "")
            # Links to sub-pages with more detail
            if href.startswith("/") and href.count("/") >= 3 and href != path:
                article_links.add(href)

        # Extract descriptions from page body text blocks
        # vietnam.travel uses .desc, .info-content, .wrap-content classes
        desc_candidates = []
        for sel in [".desc", ".info-content p", ".wrap-content p",
                    ".col-md-9 p", ".body-text", "article p", ".content p"]:
            els = page.css(sel)
            for el in els[:5]:
                text = el.get_all_text(strip=True)
                if len(text) > 80:
                    desc_candidates.append(text)

        description = " ".join(desc_candidates[:2])[:600] if desc_candidates else ""

        # Extract place names from headers / titles
        for sel in ["h1", "h2", "h3", ".title", "[class*='title']"]:
            els = page.css(sel)
            for el in els[:10]:
                name = el.get_all_text(strip=True)
                if 5 < len(name) < 120 and not name.startswith("http"):
                    records.append({
                        "place_name":  name,
                        "province":    province,
                        "description": description[:300],
                        "source_url":  url,
                        "source":      "vietnam.travel",
                    })
                    break  # one name per selector level

        # Also record the city-level description
        if description and province not in [r["place_name"] for r in records]:
            records.insert(0, {
                "place_name":  province,
                "province":    province,
                "description": description[:400],
                "source_url":  url,
                "source":      "vietnam.travel",
            })

        log.info(f"  {province}: {len(records)} entries, {len(article_links)} sub-links")

        # Scrape up to 3 sub-article pages
        scraped_sub = 0
        for sub_path in list(article_links)[:3]:
            sub_url = VT_BASE + sub_path
            try:
                sub_page = Fetcher.get(sub_url, impersonate="chrome", timeout=15)
                if sub_page.status != 200:
                    continue

                title = (sub_page.css("h1::text").get() or
                         sub_page.css(".title::text").get() or "").strip()
                if not title or len(title) < 5:
                    continue

                paras = sub_page.css("p::text, .desc::text, .body-text::text").getall()
                desc = " ".join(p.strip() for p in paras if len(p.strip()) > 30)[:500]

                records.append({
                    "place_name":  title,
                    "province":    province,
                    "description": desc,
                    "source_url":  sub_url,
                    "source":      "vietnam.travel",
                })
                scraped_sub += 1
                time.sleep(0.6)
            except Exception:
                pass

        if scraped_sub:
            log.info(f"    + {scraped_sub} sub-articles")

    except Exception as e:
        log.error(f"  {province}: {e}")

    return records


def scrape_things_to_do(url: str) -> list[dict]:
    """Scrape vietnam.travel things-to-do category pages for article links."""
    from scrapling.fetchers import Fetcher

    records = []
    try:
        page = Fetcher.get(url, impersonate="chrome", timeout=20)
        if page.status != 200:
            return records

        # Find article cards/links
        for a in page.css("a[href]"):
            href = a.attrib.get("href", "")
            text = a.get_all_text(strip=True)
            if (href.startswith("/things-to-do/") or href.startswith("/place")) \
                    and len(text) > 5 and len(text) < 150:
                records.append({
                    "place_name":  text[:120],
                    "province":    "",
                    "description": "",
                    "source_url":  VT_BASE + href if href.startswith("/") else href,
                    "source":      "vietnam.travel",
                })

        log.info(f"  things-to-do ({url.split('/')[-1]}): {len(records)} links")
    except Exception as e:
        log.error(f"  {url}: {e}")

    return records


def enrich_places_with_descriptions(scraped: pd.DataFrame) -> None:
    """Add description column to vn_tourist_places.csv using scraped data."""
    places_path = os.path.join(FEAT_DIR, "vn_tourist_places.csv")
    if not os.path.exists(places_path):
        log.warning("vn_tourist_places.csv not found")
        return

    places = pd.read_csv(places_path)

    # Add description column if not present
    if "description" not in places.columns:
        places["description"] = ""

    try:
        from rapidfuzz import process, fuzz
    except ImportError:
        log.warning("rapidfuzz not installed — skipping name-match enrichment")
        return

    enriched = 0
    scraped_with_desc = scraped[scraped["description"].str.len() > 20]

    for _, row in scraped_with_desc.iterrows():
        if not row["description"]:
            continue

        # Filter by province if available
        if row["province"]:
            pool = places[places["province"].str.contains(
                row["province"][:5], case=False, na=False)]
        else:
            pool = places

        if pool.empty:
            pool = places

        match = process.extractOne(
            row["place_name"],
            pool["place_name"].tolist(),
            scorer=fuzz.token_sort_ratio,
            score_cutoff=65,
        )
        if not match:
            continue

        idx = pool[pool["place_name"] == match[0]].index
        if idx.empty:
            continue
        i = idx[0]

        if not places.at[i, "description"] or len(str(places.at[i, "description"])) < 30:
            places.at[i, "description"] = row["description"][:400]
            enriched += 1

    places.to_csv(places_path, index=False)
    has_desc = places["description"].str.len().gt(20).sum()
    log.info(f"Places enriched: {enriched} descriptions added ({has_desc}/{len(places)} total)")


def main():
    all_records = []

    # Scrape city pages
    log.info("=== Scraping vietnam.travel city pages ===")
    for province, path in CITY_PAGES:
        records = scrape_city_page(province, path)
        all_records.extend(records)
        time.sleep(0.8)

    # Scrape things-to-do pages
    log.info("\n=== Scraping vietnam.travel things-to-do ===")
    for url in THINGS_TO_DO_PAGES:
        records = scrape_things_to_do(url)
        all_records.extend(records)
        time.sleep(0.5)

    if not all_records:
        log.error("No data scraped.")
        return

    df = pd.DataFrame(all_records).drop_duplicates(subset=["place_name"]).reset_index(drop=True)
    out_path = os.path.join(OUT_DIR, "vn_places_detail.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    log.info(f"\nSaved → {out_path} ({len(df)} rows)")

    has_desc = df["description"].str.len().gt(20).sum()
    print(f"\nPlaces scraped       : {len(df)}")
    print(f"With description     : {has_desc} ({has_desc/len(df)*100:.1f}%)")
    print(f"\nProvince breakdown:")
    prov = df[df["province"] != ""].groupby("province").size().sort_values(ascending=False)
    print(prov.head(12).to_string())

    enrich_places_with_descriptions(df)


if __name__ == "__main__":
    main()
