"""
scrape_hotel_prices.py — Scrape hotel prices (VND/night) from Agoda Vietnam.

Uses StealthyFetcher with check-in date to get actual "Avg price per night".
Extracts: name, district, rating, review_count, avg_price_vnd, price_tier.
Matches to vn_hotels.csv by fuzzy name and merges price + rating.

Usage:
    python scripts/scrape_hotel_prices.py

Output:
    data/scraped/agoda_hotel_prices_vn.csv
    data/features/vn_hotels.csv  (updated: price_tier, estimated_price_vnd, rating)

Requirements:
    pip install "scrapling[fetchers]" rapidfuzz
"""

import os, re, time, logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE_DIR, "data", "scraped")
FEAT_DIR = os.path.join(BASE_DIR, "data", "features")
os.makedirs(OUT_DIR, exist_ok=True)

# Check-in date for price lookup
CHECKIN  = "2026-05-10"
CHECKOUT_LOS = "los=1"   # 1 night

# Agoda city URLs — using /city/ format (confirmed working)
AGODA_CITIES = [
    ("Ha Noi",      "https://www.agoda.com/city/hanoi-vn.html"),
    ("Ho Chi Minh", "https://www.agoda.com/city/ho-chi-minh-city-vn.html"),
    ("Da Nang",     "https://www.agoda.com/city/da-nang-vn.html"),
    ("Hue",         "https://www.agoda.com/city/hue-vn.html"),
    ("Khanh Hoa",   "https://www.agoda.com/city/nha-trang-vn.html"),
    ("Quang Ninh",  "https://www.agoda.com/city/halong-vn.html"),
    ("Quang Nam",   "https://www.agoda.com/city/hoi-an-vn.html"),
    ("Ninh Binh",   "https://www.agoda.com/city/ninh-binh-vn.html"),
    ("Lao Cai",     "https://www.agoda.com/city/sapa-vn.html"),
    ("Binh Thuan",  "https://www.agoda.com/city/mui-ne-vn.html"),
]

# VND price → tier thresholds
def vnd_to_tier(price_vnd: float) -> tuple[str, float]:
    if price_vnd < 500_000:
        return "budget", price_vnd
    elif price_vnd < 1_500_000:
        return "mid_range", price_vnd
    elif price_vnd < 4_000_000:
        return "premium", price_vnd
    else:
        return "luxury", price_vnd


def parse_vnd_price(text: str) -> float | None:
    """Parse '₫ 148,919' or '₫148919' → 148919.0"""
    m = re.search(r"₫\s*([\d,\.]+)", text.replace("\u200b", ""))
    if m:
        try:
            return float(m.group(1).replace(",", "").replace(".", ""))
        except ValueError:
            pass
    return None


def parse_rating_and_reviews(lines: list[str]) -> tuple[float | None, int | None]:
    score, count = None, None
    for line in lines:
        if score is None:
            m = re.search(r"\b(\d\.\d)\b", line)
            if m:
                try:
                    score = float(m.group(1))
                except ValueError:
                    pass
        if count is None:
            m2 = re.search(r"(\d[\d,]*)\s*(?:reviews?|nhận xét|đánh giá)", line, re.IGNORECASE)
            if m2:
                try:
                    count = int(m2.group(1).replace(",", ""))
                except ValueError:
                    pass
    return score, count


def scrape_city(province: str, base_url: str) -> list[dict]:
    from scrapling.fetchers import StealthyFetcher

    records = []
    url = f"{base_url}?checkIn={CHECKIN}&{CHECKOUT_LOS}&adults=2&rooms=1"
    log.info(f"  {province}: {url}")

    try:
        page = StealthyFetcher.fetch(
            url,
            headless=True,
            solve_cloudflare=False,
            network_idle=True,
            timeout=55000,
        )
        if page.status not in (200,):
            log.warning(f"    HTTP {page.status}")
            return records

        # Property cards — confirmed selector from DOM inspection
        cards = page.css("[class*='DatelessPropertyCard']")
        if not cards:
            cards = page.css("[class*='PropertyCard']")
        log.info(f"    {len(cards)} card elements")

        UI_SKIP = {"see all", "view on map", "xem trên bản đồ", "free wi-fi",
                   "miễn phí wi-fi", "đặt ngay", "avg price per night",
                   "avg price", "tìm kiếm", "check in", "check out"}

        seen_names = set()
        for card in cards:
            text = card.get_all_text(strip=True)
            if not text or len(text) < 20:
                continue
            lines = [l.strip() for l in text.split('\n') if l.strip()]

            # Extract name — first substantial line not in UI_SKIP
            name = ""
            for line in lines:
                if len(line) > 8 and line.lower() not in UI_SKIP and not line.startswith('"'):
                    name = re.sub(r"\s*\([^)]+\)\s*$", "", line).strip()
                    break
            if not name or name in seen_names:
                continue
            seen_names.add(name)

            # Price
            price_vnd = None
            price_lines = [l for l in lines if "₫" in l]
            for pl in price_lines:
                price_vnd = parse_vnd_price(pl)
                if price_vnd:
                    break

            # Rating + reviews
            score, rev_count = parse_rating_and_reviews(lines)

            # District
            district = ""
            for line in lines:
                if " - " in line and any(kw in line.lower() for kw in
                    ["hanoi","ho chi minh","da nang","hue","nha trang","ha long",
                     "hoi an","ninh binh","sapa","mui ne","hà nội","đà nẵng"]):
                    district = line.split(" - ")[0].strip()
                    break

            tier, _ = vnd_to_tier(price_vnd) if price_vnd else ("unknown", 0)

            records.append({
                "name":               name,
                "province":           province,
                "district":           district,
                "price_vnd_night":    price_vnd,
                "price_tier":         tier,
                "rating_score":       score,
                "review_count":       rev_count,
                "source":             "agoda.com",
            })

        log.info(f"    → {len(records)} hotels, "
                 f"{sum(1 for r in records if r['price_vnd_night'])} with price")

    except Exception as e:
        log.error(f"    {province}: {e}")

    return records


def merge_into_hotels(scraped: pd.DataFrame) -> None:
    hotels_path = os.path.join(FEAT_DIR, "vn_hotels.csv")
    if not os.path.exists(hotels_path):
        return

    try:
        from rapidfuzz import process, fuzz
    except ImportError:
        log.warning("rapidfuzz not found — pip install rapidfuzz")
        return

    hotels = pd.read_csv(hotels_path)
    price_updated, rating_updated = 0, 0

    for _, row in scraped.iterrows():
        prov_hotels = hotels[hotels["province"] == row["province"]]
        if prov_hotels.empty:
            continue

        match = process.extractOne(
            row["name"],
            prov_hotels["name"].tolist(),
            scorer=fuzz.token_sort_ratio,
            score_cutoff=70,
        )
        if not match:
            continue

        idx = prov_hotels[prov_hotels["name"] == match[0]].index
        if idx.empty:
            continue
        i = idx[0]

        if row["price_vnd_night"] and row["price_tier"] != "unknown":
            hotels.at[i, "price_tier"] = row["price_tier"]
            hotels.at[i, "estimated_price_vnd"] = row["price_vnd_night"]
            price_updated += 1

        if row["rating_score"] and pd.isna(hotels.at[i, "rating"]):
            hotels.at[i, "rating"] = row["rating_score"]
            rating_updated += 1

        if row["review_count"] and pd.isna(hotels.at[i, "review_count"]):
            hotels.at[i, "review_count"] = row["review_count"]

    hotels.to_csv(hotels_path, index=False)
    log.info(f"Merged: {price_updated} price updates, {rating_updated} rating updates")
    log.info(f"Hotels with rating: {hotels['rating'].notna().sum()}/{len(hotels)}")


def main():
    all_records = []
    for province, url in AGODA_CITIES:
        records = scrape_city(province, url)
        all_records.extend(records)
        time.sleep(2.0)

    if not all_records:
        log.error("No data scraped.")
        return

    df = pd.DataFrame(all_records)
    out_path = os.path.join(OUT_DIR, "agoda_hotel_prices_vn.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    log.info(f"Saved → {out_path} ({len(df)} rows)")

    has_price = df["price_vnd_night"].notna()
    print(f"\nTotal hotels scraped : {len(df)}")
    print(f"With price           : {has_price.sum()} ({has_price.mean()*100:.1f}%)")
    print(f"With rating          : {df['rating_score'].notna().sum()}")
    if has_price.any():
        priced = df[has_price]
        print(f"\nPrice stats (VND/night):")
        print(f"  min  : {priced['price_vnd_night'].min():>12,.0f}")
        print(f"  mean : {priced['price_vnd_night'].mean():>12,.0f}")
        print(f"  max  : {priced['price_vnd_night'].max():>12,.0f}")
        print(f"\nPrice tier distribution:")
        print(df["price_tier"].value_counts().to_string())

    merge_into_hotels(df)


if __name__ == "__main__":
    main()
