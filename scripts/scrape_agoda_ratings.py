"""
scrape_agoda_ratings.py — Scrape hotel names, ratings, review counts from Agoda VN.

Uses StealthyFetcher (Camoufox) to render Agoda city listing pages.
Extracts: name, district, rating_score, review_count, amenities.
Matches scraped hotels to vn_hotels.csv by fuzzy name matching and enriches
the rating/review_count columns.

Usage:
    python scripts/scrape_agoda_ratings.py

Output:
    data/scraped/agoda_hotels_vn.csv        — raw scraped data
    data/features/vn_hotels.csv             — updated with rating/review_count

Requirements:
    pip install "scrapling[fetchers]" rapidfuzz
"""

import os, sys, re, time, logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE_DIR, "data", "scraped")
FEAT_DIR = os.path.join(BASE_DIR, "data", "features")
os.makedirs(OUT_DIR, exist_ok=True)

# Agoda VN city pages (dateless — shows rating without needing dates)
AGODA_CITIES = [
    ("Ha Noi",      "https://www.agoda.com/vi-vn/city/hanoi-vn.html"),
    ("Ho Chi Minh", "https://www.agoda.com/vi-vn/city/ho-chi-minh-city-vn.html"),
    ("Da Nang",     "https://www.agoda.com/vi-vn/city/da-nang-vn.html"),
    ("Hue",         "https://www.agoda.com/vi-vn/city/hue-vn.html"),
    ("Khanh Hoa",   "https://www.agoda.com/vi-vn/city/nha-trang-vn.html"),
    ("Lam Dong",    "https://www.agoda.com/vi-vn/city/da-lat-vn.html"),
    ("Quang Ninh",  "https://www.agoda.com/vi-vn/city/ha-long-vn.html"),
    ("Kien Giang",  "https://www.agoda.com/vi-vn/city/phu-quoc-vn.html"),
    ("Quang Nam",   "https://www.agoda.com/vi-vn/city/hoi-an-vn.html"),
    ("Ninh Binh",   "https://www.agoda.com/vi-vn/city/ninh-binh-vn.html"),
]


def parse_rating(text: str) -> tuple[float | None, int | None]:
    """Extract (score, review_count) from text like '8.6 Tuyệt vời 199 nhận xét'."""
    score = None
    count = None
    # Score: number like "8.6" or "9.2"
    m = re.search(r"\b(\d\.\d)\b", text)
    if m:
        try:
            score = float(m.group(1))
        except ValueError:
            pass
    # Review count: number before "nhận xét" or "reviews" or "đánh giá"
    m2 = re.search(r"(\d[\d,\.]*)\s*(?:nhận xét|reviews|đánh giá|reviews?)", text)
    if m2:
        try:
            count = int(m2.group(1).replace(",", "").replace(".", ""))
        except ValueError:
            pass
    return score, count


def scrape_city(province: str, url: str) -> list[dict]:
    """Scrape hotel listing page for one province."""
    from scrapling.fetchers import StealthyFetcher

    records = []
    log.info(f"  Scraping {province}: {url}")
    try:
        page = StealthyFetcher.fetch(
            url,
            headless=True,
            solve_cloudflare=False,
            network_idle=True,
            timeout=55000,
        )
        if page.status not in (200, 302):
            log.warning(f"    HTTP {page.status}")
            return records

        # Find property cards — Agoda uses DatelessPropertyCard at the top level
        # We need the card that contains BOTH name AND rating
        all_cards = page.css("[class*='DatelessPropertyCard']:not([class*='Gallery']):not([class*='Hero']):not([class*='Thumbnail'])")

        # If dedicated selector fails, fall back to broader approach
        if not all_cards:
            all_cards = page.css("[class*='PropertyCard']")

        log.info(f"    Found {len(all_cards)} raw card elements")

        seen_names = set()
        for card in all_cards:
            text = card.get_all_text(strip=True)
            if not text or len(text) < 20:
                continue

            lines = [l.strip() for l in text.split('\n') if l.strip()]
            if not lines:
                continue

            # Name: usually the first substantial line, in Vietnamese or English
            # Skip lines that are navigation/UI text
            UI_SKIP = {"xem hết", "xem trên bản đồ", "miễn phí wi-fi", "tìm",
                       "đăng nhập", "đặt ngay", "kiểm tra"}
            name = ""
            for line in lines:
                if len(line) > 8 and line.lower() not in UI_SKIP and not line.startswith('"'):
                    # Remove parenthetical English duplicate  "(Name English)"
                    name = re.sub(r"\s*\([^)]+\)\s*$", "", line).strip()
                    break

            if not name or name in seen_names:
                continue
            seen_names.add(name)

            # Rating + review count
            score, count = None, None
            for line in lines:
                s, c = parse_rating(line)
                if s is not None:
                    score = s
                if c is not None:
                    count = c
                if score and count:
                    break

            # District / area (line with "Hà Nội - Xem trên bản đồ")
            district = ""
            for line in lines:
                if " - " in line and any(kw in line for kw in ["Hà Nội", "Hồ Chí", "Đà Nẵng",
                                                                "Huế", "Nha Trang", "Đà Lạt",
                                                                "Hạ Long", "Phú Quốc", "Hội An"]):
                    district = line.split(" - ")[0].strip()
                    break

            records.append({
                "name":         name,
                "province":     province,
                "district":     district,
                "rating_score": score,
                "review_count": count,
                "source":       "agoda.com",
            })

        log.info(f"    Extracted {len(records)} hotels for {province}")

    except Exception as e:
        log.error(f"    Error for {province}: {e}")

    return records


def merge_ratings_into_hotels(scraped: pd.DataFrame) -> None:
    """Fuzzy-match scraped Agoda hotels to OSM hotels and fill in rating/review_count."""
    hotels_path = os.path.join(FEAT_DIR, "vn_hotels.csv")
    if not os.path.exists(hotels_path):
        log.warning("vn_hotels.csv not found")
        return

    try:
        from rapidfuzz import process, fuzz
    except ImportError:
        log.warning("rapidfuzz not installed (pip install rapidfuzz). Skipping merge.")
        return

    hotels = pd.read_csv(hotels_path)
    updated = 0

    for _, row in scraped.iterrows():
        if row["rating_score"] is None and row["review_count"] is None:
            continue

        province_hotels = hotels[hotels["province"] == row["province"]]
        if province_hotels.empty:
            continue

        match = process.extractOne(
            row["name"],
            province_hotels["name"].tolist(),
            scorer=fuzz.token_sort_ratio,
            score_cutoff=72,
        )
        if not match:
            continue

        idx = province_hotels[province_hotels["name"] == match[0]].index
        if idx.empty:
            continue
        i = idx[0]

        if row["rating_score"] is not None and pd.isna(hotels.at[i, "rating"]):
            hotels.at[i, "rating"] = row["rating_score"]
            updated += 1
        if row["review_count"] is not None and pd.isna(hotels.at[i, "review_count"]):
            hotels.at[i, "review_count"] = row["review_count"]

    hotels.to_csv(hotels_path, index=False)
    filled_rating = hotels["rating"].notna().sum()
    log.info(f"Merge: {updated} hotels updated, {filled_rating}/{len(hotels)} now have ratings")


def main():
    all_records = []

    for province, url in AGODA_CITIES:
        records = scrape_city(province, url)
        all_records.extend(records)
        log.info(f"  {province}: {len(records)} hotels total so far {len(all_records)}")
        time.sleep(2.5)  # polite delay

    if not all_records:
        log.error("No hotel data scraped.")
        return

    df = pd.DataFrame(all_records)
    out_path = os.path.join(OUT_DIR, "agoda_hotels_vn.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    log.info(f"Saved → {out_path} ({len(df)} rows)")

    # Summary
    print(f"\nTotal hotels scraped: {len(df)}")
    print(f"With rating:         {df['rating_score'].notna().sum()}")
    print(f"With review count:   {df['review_count'].notna().sum()}")
    print(f"\nProvince breakdown:")
    print(df.groupby("province")[["rating_score"]].count().rename(
        columns={"rating_score": "hotels"}).to_string())

    if df["rating_score"].notna().any():
        print(f"\nRating stats: mean={df['rating_score'].mean():.2f}, "
              f"min={df['rating_score'].min()}, max={df['rating_score'].max()}")

    # Merge into features
    merge_ratings_into_hotels(df)


if __name__ == "__main__":
    main()
