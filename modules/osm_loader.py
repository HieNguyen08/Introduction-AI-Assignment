"""
osm_loader.py — Load Vietnam tourist data from the pre-extracted OSM JSON.

Source: data/osm/pbf_asia.json  (produced from asia-260327.osm.pbf)
Contains: 10,221 VN restaurants, 2,630 VN hotels, 1,466 VN attractions.

All three loaders produce DataFrames with a unified schema that includes
rating, review_count, and description columns (empty from OSM, enriched
later by web scraping scripts in scripts/scrape_*.py).

Usage:
    from modules.osm_loader import load_osm_places, load_osm_restaurants, load_osm_hotels
"""

import os
import re as _re
import json
import logging
import pandas as pd
from typing import Optional

logger = logging.getLogger("osm_loader")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OSM_JSON = os.path.join(BASE_DIR, "data", "osm", "pbf_asia.json")

# ─────────────────────────────────────────────────────────────────────────────
# OSM type → project category (5 buckets)
# ─────────────────────────────────────────────────────────────────────────────
TYPE_TO_CATEGORY: dict[str, str] = {
    # culture
    "museum":                       "culture",
    "gallery":                      "culture",
    "historic_monument":            "culture",
    "historic_memorial":            "culture",
    "historic_ruins":               "culture",
    "historic_archaeological_site": "culture",
    "historic_castle":              "culture",
    "historic_city_gate":           "culture",
    "historic_wayside_shrine":      "culture",
    "cultural_theatre":             "culture",
    "cultural_arts_centre":         "culture",
    "historic_temple":              "culture",
    "historic_pagoda":              "culture",
    "place_of_worship":             "culture",
    "attraction":                   "culture",   # generic — filtered further
    # nature
    "viewpoint":                    "nature",
    "park":                         "nature",
    "garden":                       "nature",
    "nature_reserve":               "nature",
    "waterfall":                    "nature",
    "cave_entrance":                "adventure",
    "peak":                         "adventure",
    # beach
    "beach":                        "beach",
    "beach_resort":                 "beach",
    # entertainment
    "water_park":                   "entertainment",
    "theme_park":                   "entertainment",
    "aquarium":                     "entertainment",
    "zoo":                          "entertainment",
    "cultural_cinema":              "entertainment",
    "cultural_nightclub":           "entertainment",
    "stadium":                      "entertainment",
    "cultural_community_centre":    "entertainment",
}

# Skip entirely — non-destination OSM tags
SKIP_TYPES = {
    "artwork", "information", "cultural_library", "cultural_casino",
    "sports_centre", "marina", "picnic_site", "caravan_site", "camp_site",
}

# ─────────────────────────────────────────────────────────────────────────────
# City → province mapping
# ─────────────────────────────────────────────────────────────────────────────
CITY_TO_PROVINCE: dict[str, str] = {
    "hanoi":                "Ha Noi",
    "ha noi":               "Ha Noi",
    "hà nội":               "Ha Noi",
    "ho chi minh city":     "Ho Chi Minh",
    "ho chi minh":          "Ho Chi Minh",
    "hồ chí minh":          "Ho Chi Minh",
    "saigon":               "Ho Chi Minh",
    "da nang":              "Da Nang",
    "đà nẵng":              "Da Nang",
    "danang":               "Da Nang",
    "hue":                  "Hue",
    "huế":                  "Hue",
    "hội an":               "Quang Nam",
    "hoi an":               "Quang Nam",
    "nha trang":            "Khanh Hoa",
    "nha trang city":       "Khanh Hoa",
    "da lat":               "Lam Dong",
    "đà lạt":               "Lam Dong",
    "dalat":                "Lam Dong",
    "hai phong":            "Hai Phong",
    "hải phòng":            "Hai Phong",
    "can tho":              "Can Tho",
    "cần thơ":              "Can Tho",
    "vung tau":             "Ba Ria Vung Tau",
    "vũng tàu":             "Ba Ria Vung Tau",
    "quy nhon":             "Binh Dinh",
    "qui nhon":             "Binh Dinh",
    "phu quoc":             "Kien Giang",
    "phú quốc":             "Kien Giang",
    "ha long":              "Quang Ninh",
    "hạ long":              "Quang Ninh",
    "ninh binh":            "Ninh Binh",
    "ninh bình":            "Ninh Binh",
    "sapa":                 "Lao Cai",
    "sa pa":                "Lao Cai",
    "lao cai":              "Lao Cai",
    "dong hoi":             "Quang Binh",
    "đồng hới":             "Quang Binh",
    "thanh hoa":            "Thanh Hoa",
    "thanh hoá":            "Thanh Hoa",
    "vinh":                 "Nghe An",
    "ha giang":             "Ha Giang",
    "hà giang":             "Ha Giang",
    "cao bang":             "Cao Bang",
    "cao bằng":             "Cao Bang",
    "mui ne":               "Binh Thuan",
    "mũi né":               "Binh Thuan",
    "buon ma thuot":        "Dak Lak",
    "buôn ma thuột":        "Dak Lak",
    "pleiku":               "Gia Lai",
    "kon tum":              "Kon Tum",
    "tam ky":               "Quang Nam",
    "quang ngai":           "Quang Ngai",
    "dong ha":              "Quang Tri",
    "hue city":             "Hue",
}

# Province centroids for nearest-province fallback
_PROVINCE_CENTROIDS = {
    "Ha Noi":          (21.0285, 105.8542),
    "Ho Chi Minh":     (10.8231, 106.6297),
    "Da Nang":         (16.0544, 108.2022),
    "Hue":             (16.4637, 107.5909),
    "Khanh Hoa":       (12.2388, 109.1967),
    "Lam Dong":        (11.9404, 108.4583),
    "Hai Phong":       (20.8449, 106.6881),
    "Quang Ninh":      (21.0064, 107.2925),
    "Lao Cai":         (22.4856, 103.9707),
    "Ninh Binh":       (20.2506, 105.9745),
    "Quang Nam":       (15.5394, 108.0191),
    "Binh Thuan":      (11.0904, 108.0721),
    "Kien Giang":      (9.8250,  105.1259),
    "Quang Binh":      (17.4690, 106.6222),
    "Ha Giang":        (22.8233, 104.9836),
    "Cao Bang":        (22.6657, 106.2657),
    "Thanh Hoa":       (19.8067, 105.7852),
    "Nghe An":         (18.6783, 105.6813),
    "Binh Dinh":       (13.7830, 109.2197),
    "Ba Ria Vung Tau": (10.5417, 107.2429),
    "Son La":          (21.3270, 103.9144),
    "Can Tho":         (10.0452, 105.7469),
    "Binh Duong":      (11.3254, 106.4770),
    "Dong Nai":        (11.0686, 107.1676),
    "An Giang":        (10.3899, 105.4353),
    "Gia Lai":         (13.9833, 108.0000),
    "Kon Tum":         (14.3498, 108.0004),
    "Dak Lak":         (12.7100, 108.2378),
    "Quang Ngai":      (15.1207, 108.8044),
    "Quang Tri":       (16.7500, 107.1854),
}


def _nearest_province(lat: float, lon: float) -> str:
    best, best_d = "Unknown", float("inf")
    for prov, (plat, plon) in _PROVINCE_CENTROIDS.items():
        d = (lat - plat) ** 2 + (lon - plon) ** 2
        if d < best_d:
            best_d, best = d, prov
    return best


def _resolve_province(city: str, lat: float, lon: float) -> str:
    province = CITY_TO_PROVINCE.get(city.lower().strip())
    if province:
        return province
    return _nearest_province(lat, lon)


def _quality_name(row: dict) -> str:
    """Pick the best available name: prefer English if it looks English."""
    name_en = (row.get("name_en") or "").strip()
    name    = (row.get("name")    or "").strip()
    if name_en and name_en != name:
        ascii_ratio = sum(1 for c in name_en if ord(c) < 128) / max(len(name_en), 1)
        if ascii_ratio >= 0.80:
            return name_en
    return name_en or name


# ─────────────────────────────────────────────────────────────────────────────
# Opening-hours parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_opening_hours(raw: str) -> tuple[int, int] | None:
    """
    Parse OSM opening_hours string → (open_hour, close_hour).

    Handles:  "08:00-22:00"  |  "Mo-Su 07:00-23:00"  |  "24/7"
    Returns None if unparseable.
    """
    if not raw:
        return None
    raw = raw.strip()
    if raw in ("24/7", "24/7;", "always", "open"):
        return (0, 24)
    m = _re.search(r"(\d{1,2}):(\d{2})\s*[-–]\s*(\d{1,2}):(\d{2})", raw)
    if m:
        open_h, close_h = int(m.group(1)), int(m.group(3))
        if 0 <= open_h <= 23 and 1 <= close_h <= 24:
            return (open_h, close_h)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Cuisine normalisation (expanded)
# ─────────────────────────────────────────────────────────────────────────────
_CUISINE_MAP = {
    # Vietnamese
    "vietnamese": "vietnamese", "pho": "vietnamese", "bun": "vietnamese",
    "banh_mi": "vietnamese", "com": "vietnamese", "banh": "vietnamese",
    "regional": "vietnamese", "local": "vietnamese",
    # Seafood
    "seafood": "seafood", "fish": "seafood", "fish_and_chips": "seafood",
    "fish_and_rice": "seafood", "shellfish": "seafood",
    # Japanese
    "japanese": "japanese", "sushi": "japanese", "ramen": "japanese",
    "tempura": "japanese", "teppanyaki": "japanese",
    # Chinese
    "chinese": "chinese", "dim_sum": "chinese", "cantonese": "chinese",
    "hotpot": "chinese",
    # Korean
    "korean": "korean", "korean_bbq": "korean",
    # Asian (mixed / Thai / SE Asian)
    "asian": "asian", "thai": "asian", "vietnamese;asian": "vietnamese",
    "pan_asian": "asian", "indian": "asian",
    # Fast food / Western
    "burger": "fast_food", "pizza": "western", "italian": "western",
    "american": "fast_food", "fast_food": "fast_food", "sandwich": "fast_food",
    "steak": "western", "french": "western", "western": "western",
    "chicken": "fast_food", "fried_chicken": "fast_food",
    "beefsteak": "western", "noodle": "vietnamese",
    "pancake": "western", "crepe": "western",
    # BBQ / Grill
    "bbq": "bbq", "grill": "bbq", "barbecue": "bbq",
    # Cafe / Drinks
    "coffee_shop": "cafe", "cafe": "cafe", "tea": "cafe",
    "bubble_tea": "cafe", "ice_cream": "cafe", "juice": "cafe",
    "bakery": "cafe", "dessert": "cafe", "coffee": "cafe",
    # Breakfast
    "breakfast": "cafe", "brunch": "cafe",
    # Rice / Generic
    "rice": "vietnamese",
}

def _normalise_cuisine(raw_list: list) -> str:
    """Map a list of OSM cuisine tags to one of our cuisine buckets."""
    if not raw_list:
        return "unknown"
    for item in raw_list:
        tag = str(item).lower().strip().replace(" ", "_").replace(";", "")
        if tag in _CUISINE_MAP:
            return _CUISINE_MAP[tag]
        # Partial match for compound tags like "pizza;pasta"
        for sub in tag.split(";"):
            sub = sub.strip()
            if sub in _CUISINE_MAP:
                return _CUISINE_MAP[sub]
    return str(raw_list[0]).lower()[:30] if raw_list else "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Hotel price estimation
# ─────────────────────────────────────────────────────────────────────────────
_STAR_TO_TIER = {1: "budget", 2: "budget", 3: "mid_range", 4: "premium", 5: "luxury"}
_PROP_TO_TIER = {
    "HOSTEL":      "budget",
    "GUEST_HOUSE": "budget",
    "MOTEL":       "budget",
    "APARTMENT":   "mid_range",
    "HOTEL":       "mid_range",
    "RESORT":      "premium",
    "VILLA":       "premium",
}
_TIER_PRICE_VND = {
    "budget":    350_000,
    "mid_range": 1_200_000,
    "premium":   3_000_000,
    "luxury":    6_000_000,
    "unknown":   800_000,
}

# Name-based luxury/premium signals
_LUXURY_KEYWORDS = {"intercontinental", "hilton", "marriott", "sheraton", "hyatt",
                    "sofitel", "novotel", "pullman", "four seasons", "park hyatt",
                    "jw marriott", "ritz", "renaissance", "melia", "grand"}
_BUDGET_KEYWORDS = {"hostel", "backpacker", "dorm", "homestay", "guesthouse", "nhà trọ"}


def _estimate_price_tier(star: Optional[int], prop_type: str, name: str) -> str:
    """Estimate price tier from star rating, property type, and name keywords."""
    name_lower = name.lower()
    if any(kw in name_lower for kw in _LUXURY_KEYWORDS):
        return "luxury"
    if any(kw in name_lower for kw in _BUDGET_KEYWORDS):
        return "budget"
    if star is not None:
        return _STAR_TO_TIER.get(star, "mid_range")
    return _PROP_TO_TIER.get(prop_type.upper(), "mid_range")


# ─────────────────────────────────────────────────────────────────────────────
# Places loader
# ─────────────────────────────────────────────────────────────────────────────

def load_osm_places(
    json_path: Optional[str] = None,
    min_name_len: int = 5,
    max_per_province: int = 100,
) -> Optional[pd.DataFrame]:
    """
    Load Vietnam tourist attractions from pbf_asia.json.

    Returns DataFrame with schema compatible with build_places_dataframe():
        place_name, latitude, longitude, category, province,
        entry_fee_vnd, visit_duration_hours, opening_hour, closing_hour
    """
    path = json_path or OSM_JSON
    if not os.path.exists(path):
        logger.warning("OSM JSON not found: %s", path)
        return None

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    vn_raw = [a for a in raw.get("attractions", []) if a.get("country_code") == "VN"]
    logger.info("Vietnam attractions in OSM JSON: %d", len(vn_raw))

    rows = []
    for a in vn_raw:
        osm_type = a.get("type", "")
        if osm_type in SKIP_TYPES:
            continue
        category = TYPE_TO_CATEGORY.get(osm_type)
        if category is None:
            continue

        name = _quality_name(a)
        if len(name) < min_name_len:
            continue

        lat, lon = a.get("lat"), a.get("lon")
        if lat is None or lon is None:
            continue
        if not (8.18 <= lat <= 23.39 and 102.14 <= lon <= 109.46):
            continue

        province = _resolve_province(a.get("city", ""), lat, lon)

        hours = parse_opening_hours(a.get("opening_hours", ""))
        if hours:
            open_h, close_h = hours
        else:
            open_h, close_h = (9, 22) if category == "entertainment" else (7, 17)

        # Parse entry fee
        fee_raw = a.get("fee", "") or ""
        entry_fee = 0
        if fee_raw and fee_raw.lower() not in ("", "no", "free", "0"):
            m = _re.search(r"(\d[\d,\.]*)", fee_raw.replace(",", ""))
            if m:
                try:
                    entry_fee = int(float(m.group(1)))
                except ValueError:
                    pass

        visit_hrs = {"nature": 3.0, "beach": 3.0, "adventure": 3.0}.get(category, 2.0)

        rows.append({
            "place_name":           name,
            "latitude":             round(lat, 6),
            "longitude":            round(lon, 6),
            "category":             category,
            "province":             province,
            "entry_fee_vnd":        entry_fee,
            "visit_duration_hours": visit_hrs,
            "opening_hour":         open_h,
            "closing_hour":         close_h,
            "source":               "osm",
        })

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No usable Vietnam attractions found in OSM JSON")
        return df

    # Dedup: same name + within ~500m
    df = df.sort_values("place_name").reset_index(drop=True)
    keep_mask = [True] * len(df)
    seen: list[tuple] = []
    for i, row in df.iterrows():
        for sname, slat, slon in seen:
            if (row["place_name"] == sname
                    and abs(row["latitude"] - slat) < 0.005
                    and abs(row["longitude"] - slon) < 0.005):
                keep_mask[i] = False
                break
        if keep_mask[i]:
            seen.append((row["place_name"], row["latitude"], row["longitude"]))
    df = df[keep_mask].reset_index(drop=True)

    if max_per_province:
        df = df.groupby("province").head(max_per_province).reset_index(drop=True)

    df = df.drop(columns="source", errors="ignore")
    logger.info("OSM places loaded: %d unique", len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Restaurant loader
# ─────────────────────────────────────────────────────────────────────────────

def load_osm_restaurants(
    json_path: Optional[str] = None,
    min_name_len: int = 3,
    max_per_province: int = 500,
) -> Optional[pd.DataFrame]:
    """
    Load Vietnam restaurants from pbf_asia.json.

    10,221 VN entries available; 38.6% have cuisine data; all have opening hours.

    Returns DataFrame:
        name, latitude, longitude, province, cuisine, price_level,
        open_hour, close_hour, takeaway, outdoor_seating, wheelchair,
        description, rating, review_count
    """
    path = json_path or OSM_JSON
    if not os.path.exists(path):
        logger.warning("OSM JSON not found: %s", path)
        return None

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    vn_raw = [r for r in raw.get("restaurants", []) if r.get("country_code") == "VN"]
    logger.info("Vietnam restaurants in OSM JSON: %d", len(vn_raw))

    rows = []
    for r in vn_raw:
        name = (r.get("name_en") or r.get("name") or "").strip()
        if len(name) < min_name_len:
            continue

        lat, lon = r.get("lat"), r.get("lon")
        if lat is None or lon is None:
            continue
        if not (8.18 <= lat <= 23.39 and 102.14 <= lon <= 109.46):
            continue

        province = _resolve_province(r.get("city", ""), lat, lon)
        hours = parse_opening_hours(r.get("opening_hours", ""))
        open_h, close_h = hours if hours else (7, 22)
        cuisine = _normalise_cuisine(r.get("cuisine") or [])

        # Price level mapping from OSM price_level (1-4 scale)
        pl_raw = r.get("price_level")
        if pl_raw in (1, "1"):
            price_level = "budget"
        elif pl_raw in (2, "2"):
            price_level = "mid"
        elif pl_raw in (3, "3", 4, "4"):
            price_level = "upscale"
        else:
            price_level = "unknown"

        rows.append({
            "name":             name,
            "latitude":         round(lat, 6),
            "longitude":        round(lon, 6),
            "province":         province,
            "cuisine":          cuisine,
            "price_level":      price_level,
            "open_hour":        open_h,
            "close_hour":       close_h,
            "takeaway":         bool(r.get("takeaway")),
            "outdoor_seating":  bool(r.get("outdoor_seating")),
            "wheelchair":       bool(r.get("wheelchair")),
            "description":      (r.get("description") or "")[:300],
            "rating":           None,       # enriched by scraping
            "review_count":     None,       # enriched by scraping
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.drop_duplicates(subset=["name", "province"]).reset_index(drop=True)
    if max_per_province:
        df = df.groupby("province").head(max_per_province).reset_index(drop=True)

    logger.info("OSM restaurants loaded: %d (VN)", len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Hotel loader
# ─────────────────────────────────────────────────────────────────────────────

def load_osm_hotels(
    json_path: Optional[str] = None,
    min_name_len: int = 4,
    max_per_province: int = 500,
) -> Optional[pd.DataFrame]:
    """
    Load Vietnam hotels from pbf_asia.json.

    2,630 VN entries; 40 have star_rating; 0 have price_per_night.
    Price tier is estimated from star_rating → property_type → name keywords.

    Returns DataFrame:
        name, latitude, longitude, province, property_type, star_rating,
        price_tier, estimated_price_vnd, wheelchair, internet_access,
        description, rating, review_count
    """
    path = json_path or OSM_JSON
    if not os.path.exists(path):
        logger.warning("OSM JSON not found: %s", path)
        return None

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    vn_raw = [h for h in raw.get("hotels", []) if h.get("country_code") == "VN"]
    logger.info("Vietnam hotels in OSM JSON: %d", len(vn_raw))

    rows = []
    for h in vn_raw:
        name = (h.get("name_en") or h.get("name") or "").strip()
        if len(name) < min_name_len:
            continue

        lat, lon = h.get("lat"), h.get("lon")
        if lat is None or lon is None:
            continue
        if not (8.18 <= lat <= 23.39 and 102.14 <= lon <= 109.46):
            continue

        province = _resolve_province(h.get("city", ""), lat, lon)

        star = h.get("star_rating")
        try:
            star = int(star) if star is not None else None
        except (TypeError, ValueError):
            star = None

        prop_type = (h.get("property_type") or "HOTEL").upper()
        tier = _estimate_price_tier(star, prop_type, name)
        estimated_price = _TIER_PRICE_VND[tier]

        rows.append({
            "name":                name,
            "latitude":            round(lat, 6),
            "longitude":           round(lon, 6),
            "province":            province,
            "property_type":       prop_type,
            "star_rating":         star,
            "price_tier":          tier,
            "estimated_price_vnd": estimated_price,
            "wheelchair":          bool(h.get("wheelchair")),
            "internet_access":     bool(h.get("internet_access")),
            "description":         (h.get("description") or "")[:300],
            "rating":              h.get("rating"),      # None for most
            "review_count":        None,                 # enriched by scraping
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.drop_duplicates(subset=["name", "province"]).reset_index(drop=True)
    if max_per_province:
        df = df.groupby("province").head(max_per_province).reset_index(drop=True)

    logger.info("OSM hotels loaded: %d (VN)", len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Merge OSM places with base VN_TOURIST_PLACES
# ─────────────────────────────────────────────────────────────────────────────

def merge_with_base(
    base_df: pd.DataFrame,
    osm_df: Optional[pd.DataFrame],
    max_osm_per_category: int = 80,
) -> pd.DataFrame:
    """
    Merge OSM attractions with curated base places (VN_TOURIST_PLACES).

    Base places take priority — OSM duplicates are dropped.
    OSM additions are capped per category to control distribution.
    """
    if osm_df is None or osm_df.empty:
        return base_df

    base_names = set(base_df["place_name"].str.lower())
    osm_new = osm_df[~osm_df["place_name"].str.lower().isin(base_names)].copy()
    osm_capped = (
        osm_new.groupby("category")
               .head(max_osm_per_category)
               .reset_index(drop=True)
    )

    combined = pd.concat([base_df, osm_capped], ignore_index=True)
    logger.info(
        "Merged: %d base + %d OSM (cap %d/cat) = %d total places",
        len(base_df), len(osm_capped), max_osm_per_category, len(combined),
    )
    return combined
