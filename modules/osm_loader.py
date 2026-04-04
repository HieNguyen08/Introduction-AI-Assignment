"""
osm_loader.py — Load Vietnam tourist places from the pre-extracted OSM JSON.

The source file (data/osm/pbf_asia.json) was produced by processing
asia-260327.osm.pbf.  This module filters it to Vietnam-only, maps OSM
types to the project's 5 categories, and returns a DataFrame with the same
schema as build_places_dataframe() in data_pipeline.py.

Usage (from data_pipeline.py):
    from modules.osm_loader import load_osm_places
    df_osm = load_osm_places()          # returns None if file not found
"""

import os
import json
import logging
import pandas as pd
from typing import Optional

logger = logging.getLogger("osm_loader")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OSM_JSON = os.path.join(BASE_DIR, "data", "osm", "pbf_asia.json")

# ---------------------------------------------------------------------------
# OSM type → project category
# ---------------------------------------------------------------------------
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
    # nature
    "viewpoint":                    "nature",
    "park":                         "nature",
    "garden":                       "nature",
    # entertainment
    "water_park":                   "entertainment",
    "theme_park":                   "entertainment",
    "aquarium":                     "entertainment",
    "cultural_cinema":              "entertainment",
    "cultural_nightclub":           "entertainment",
    "stadium":                      "entertainment",
    # generic "attraction" — mapped but quality-filtered later
    "attraction":                   "culture",
}

# Types to skip entirely
SKIP_TYPES = {
    "artwork",                  # paintings/sculptures — not visitor destinations
    "information",              # info boards
    "cultural_community_centre",
    "cultural_library",
    "cultural_casino",
    "sports_centre",
    "marina",
    "picnic_site",
    "caravan_site",
    "camp_site",
}

# City → province mapping
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
    "cao bang":             "Cao Bang",
}

# Province centroids for fallback nearest-province lookup
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
    "Thua Thien Hue":  (16.4637, 107.5909),
    "Binh Dinh":       (13.7830, 109.2197),
    "Ba Ria Vung Tau": (10.5417, 107.2429),
    "Son La":          (21.3270, 103.9144),
    "Can Tho":         (10.0452, 105.7469),
    "Binh Duong":      (11.3254, 106.4770),
    "Dong Nai":        (11.0686, 107.1676),
    "An Giang":        (10.3899, 105.4353),
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


def _looks_english(name: str) -> bool:
    """Heuristic: name is likely English if it has mostly ASCII chars."""
    if not name:
        return False
    ascii_ratio = sum(1 for c in name if ord(c) < 128) / len(name)
    return ascii_ratio >= 0.85


def _quality_name(row: dict) -> str:
    """Pick the best available name for the place."""
    name_en = (row.get("name_en") or "").strip()
    name    = (row.get("name")    or "").strip()
    # Prefer name_en if it looks English and differs from Vietnamese name
    if name_en and _looks_english(name_en) and name_en != name:
        return name_en
    # Fall back to name_en even if Vietnamese (better than empty)
    if name_en:
        return name_en
    return name


def load_osm_places(
    json_path: Optional[str] = None,
    min_name_len: int = 5,
    max_per_province: int = 30,
) -> Optional[pd.DataFrame]:
    """
    Load Vietnam tourist attractions from the pre-extracted OSM JSON.

    Args:
        json_path:        Path to pbf_asia.json (defaults to data/osm/pbf_asia.json)
        min_name_len:     Skip places with names shorter than this
        max_per_province: Cap results per province to avoid city over-representation

    Returns:
        DataFrame with columns: place_name, latitude, longitude, category,
        province, entry_fee_vnd, visit_duration_hours, opening_hour, closing_hour
        Returns None if the JSON file is not found.
    """
    path = json_path or OSM_JSON
    if not os.path.exists(path):
        logger.warning("OSM JSON not found: %s — skipping OSM enrichment", path)
        return None

    logger.info("Loading OSM places from %s", path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    attractions = raw.get("attractions", [])

    # Filter to Vietnam
    vn_raw = [a for a in attractions if a.get("country_code") == "VN"]
    logger.info("Vietnam attractions in OSM JSON: %d", len(vn_raw))

    rows = []
    for a in vn_raw:
        osm_type = a.get("type", "")

        # Skip unwanted types
        if osm_type in SKIP_TYPES:
            continue

        category = TYPE_TO_CATEGORY.get(osm_type)
        if category is None:
            continue

        name = _quality_name(a)
        if len(name) < min_name_len:
            continue

        lat = a.get("lat")
        lon = a.get("lon")
        if lat is None or lon is None:
            continue

        # Skip if outside Vietnam bounding box
        if not (8.18 <= lat <= 23.39 and 102.14 <= lon <= 109.46):
            continue

        province = _resolve_province(a.get("city", ""), lat, lon)

        # Default visit duration and hours by category
        visit_hrs = {"nature": 3.0, "beach": 3.0, "adventure": 3.0}.get(category, 2.0)
        open_h, close_h = (9, 22) if category == "entertainment" else (7, 17)

        rows.append({
            "place_name":          name,
            "latitude":            round(lat, 6),
            "longitude":           round(lon, 6),
            "category":            category,
            "province":            province,
            "entry_fee_vnd":       0,
            "visit_duration_hours": visit_hrs,
            "opening_hour":        open_h,
            "closing_hour":        close_h,
            "source":              "osm",
        })

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No usable Vietnam attractions found in OSM JSON")
        return df

    # Deduplicate: same name + within ~500m (0.005°)
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

    # Cap per province to prevent Ho Chi Minh / Hanoi dominating
    if max_per_province:
        df = (
            df.groupby("province")
              .head(max_per_province)
              .reset_index(drop=True)
        )

    df = df.drop(columns="source", errors="ignore")

    logger.info(
        "OSM places loaded: %d unique (after dedup + province cap)",
        len(df),
    )
    return df


# ---------------------------------------------------------------------------
# Opening-hours parser
# ---------------------------------------------------------------------------
import re as _re

def parse_opening_hours(raw: str) -> tuple[int, int] | None:
    """
    Parse an OSM opening_hours string into (open_hour, close_hour) ints.

    Handles common formats:
        "08:00-22:00"            → (8, 22)
        "Mo-Su 07:00-23:00"     → (7, 23)
        "24/7"                   → (0, 24)
        "Mo-Fr 09:00-17:00; Sa 10:00-14:00"  → (9, 17)  [first rule only]

    Returns None if unparseable.
    """
    if not raw:
        return None
    raw = raw.strip()
    if raw in ("24/7", "24/7;", "always"):
        return (0, 24)
    # Find HH:MM-HH:MM anywhere in the string (first occurrence)
    m = _re.search(r"(\d{1,2}):(\d{2})\s*[-–]\s*(\d{1,2}):(\d{2})", raw)
    if m:
        open_h  = int(m.group(1))
        close_h = int(m.group(3))
        if 0 <= open_h <= 23 and 1 <= close_h <= 24:
            return (open_h, close_h)
    return None


# ---------------------------------------------------------------------------
# Cuisine normalisation
# ---------------------------------------------------------------------------
_CUISINE_MAP = {
    # Vietnamese
    "vietnamese": "vietnamese", "pho": "vietnamese", "bun": "vietnamese",
    "banh_mi": "vietnamese", "com": "vietnamese",
    # Seafood
    "seafood": "seafood", "fish": "seafood", "sushi": "japanese",
    # Asian
    "japanese": "japanese", "ramen": "japanese",
    "chinese": "chinese", "korean": "korean", "thai": "asian",
    "asian": "asian",
    # Western / Fast food
    "burger": "fast_food", "pizza": "western", "italian": "western",
    "american": "fast_food", "fast_food": "fast_food", "sandwich": "fast_food",
    "steak": "western", "french": "western",
    # Cafe / Drinks
    "coffee_shop": "cafe", "cafe": "cafe", "tea": "cafe",
    "ice_cream": "cafe", "juice": "cafe",
    # BBQ / Grill
    "bbq": "bbq", "grill": "bbq", "hotpot": "bbq",
}

def _normalise_cuisine(raw_list: list) -> str:
    """Map a list of raw OSM cuisine tags to one of our cuisine buckets."""
    if not raw_list:
        return "unknown"
    for item in raw_list:
        tag = str(item).lower().replace(" ", "_")
        if tag in _CUISINE_MAP:
            return _CUISINE_MAP[tag]
    # If no known tag, return the first raw value
    return str(raw_list[0]).lower()[:20] if raw_list else "unknown"


# ---------------------------------------------------------------------------
# Restaurant loader
# ---------------------------------------------------------------------------

def load_osm_restaurants(
    json_path: Optional[str] = None,
    min_name_len: int = 3,
    max_per_province: int = 100,
) -> Optional[pd.DataFrame]:
    """
    Load Vietnam restaurants from pbf_asia.json.

    All 10,221 VN restaurant entries have opening_hours populated.
    Returns DataFrame with columns:
        name, latitude, longitude, province, cuisine, open_hour, close_hour,
        takeaway, outdoor_seating, wheelchair
    Returns None if the JSON file is not found.
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
        # Pick best available name
        name = (r.get("name_en") or r.get("name") or "").strip()
        if len(name) < min_name_len:
            continue

        lat = r.get("lat")
        lon = r.get("lon")
        if lat is None or lon is None:
            continue
        if not (8.18 <= lat <= 23.39 and 102.14 <= lon <= 109.46):
            continue

        province = _resolve_province(r.get("city", ""), lat, lon)

        hours = parse_opening_hours(r.get("opening_hours", ""))
        open_h, close_h = hours if hours else (7, 22)

        cuisine = _normalise_cuisine(r.get("cuisine") or [])

        rows.append({
            "name":             name,
            "latitude":         round(lat, 6),
            "longitude":        round(lon, 6),
            "province":         province,
            "cuisine":          cuisine,
            "open_hour":        open_h,
            "close_hour":       close_h,
            "takeaway":         bool(r.get("takeaway")),
            "outdoor_seating":  bool(r.get("outdoor_seating")),
            "wheelchair":       bool(r.get("wheelchair")),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Dedup by name + location
    df = df.drop_duplicates(subset=["name", "province"]).reset_index(drop=True)

    # Province cap
    if max_per_province:
        df = (
            df.groupby("province")
              .head(max_per_province)
              .reset_index(drop=True)
        )

    logger.info("OSM restaurants loaded: %d (VN)", len(df))
    return df


# ---------------------------------------------------------------------------
# Hotel loader
# ---------------------------------------------------------------------------

def load_osm_hotels(
    json_path: Optional[str] = None,
    min_name_len: int = 4,
    max_per_province: int = 100,
) -> Optional[pd.DataFrame]:
    """
    Load Vietnam hotels from pbf_asia.json.

    Note: star_rating populated for only ~40/2630 entries; price_per_night = 0.
    Returns DataFrame with columns:
        name, latitude, longitude, province, property_type, star_rating,
        price_tier, wheelchair, internet_access
    Returns None if the JSON file is not found.
    """
    path = json_path or OSM_JSON
    if not os.path.exists(path):
        logger.warning("OSM JSON not found: %s", path)
        return None

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    vn_raw = [h for h in raw.get("hotels", []) if h.get("country_code") == "VN"]
    logger.info("Vietnam hotels in OSM JSON: %d", len(vn_raw))

    # Estimate price tier from star_rating (stars → VND/night rough estimate)
    _STAR_TO_TIER = {1: "budget", 2: "budget", 3: "mid_range",
                     4: "premium", 5: "luxury"}
    _TIER_PRICE = {
        "budget": 500_000, "mid_range": 1_500_000,
        "premium": 3_000_000, "luxury": 6_000_000, "unknown": 1_000_000,
    }

    rows = []
    for h in vn_raw:
        name = (h.get("name_en") or h.get("name") or "").strip()
        if len(name) < min_name_len:
            continue

        lat = h.get("lat")
        lon = h.get("lon")
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

        tier = _STAR_TO_TIER.get(star, "unknown")
        estimated_price = _TIER_PRICE[tier]

        rows.append({
            "name":              name,
            "latitude":          round(lat, 6),
            "longitude":         round(lon, 6),
            "province":          province,
            "property_type":     (h.get("property_type") or "HOTEL").upper(),
            "star_rating":       star,
            "price_tier":        tier,
            "estimated_price_vnd": estimated_price,
            "wheelchair":        bool(h.get("wheelchair")),
            "internet_access":   bool(h.get("internet_access")),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Dedup + province cap
    df = df.drop_duplicates(subset=["name", "province"]).reset_index(drop=True)
    if max_per_province:
        df = (
            df.groupby("province")
              .head(max_per_province)
              .reset_index(drop=True)
        )

    logger.info("OSM hotels loaded: %d (VN)", len(df))
    return df


def merge_with_base(
    base_df: pd.DataFrame,
    osm_df: Optional[pd.DataFrame],
    max_osm_per_category: int = 20,
) -> pd.DataFrame:
    """
    Merge OSM places with the base VN_TOURIST_PLACES DataFrame.

    Base places take priority — OSM places with the same name are dropped.
    OSM additions are capped per category to prevent culture-heavy OSM data
    from skewing the overall distribution.

    Args:
        base_df:               Curated base places (VN_TOURIST_PLACES)
        osm_df:                Places from OSM JSON
        max_osm_per_category:  Max additional OSM places per category
    """
    if osm_df is None or osm_df.empty:
        return base_df

    base_names = set(base_df["place_name"].str.lower())

    # Drop OSM entries already in base
    osm_new = osm_df[~osm_df["place_name"].str.lower().isin(base_names)].copy()

    # Cap per category so the OSM bulk does not skew distribution
    osm_capped = (
        osm_new.groupby("category")
               .head(max_osm_per_category)
               .reset_index(drop=True)
    )

    combined = pd.concat([base_df, osm_capped], ignore_index=True)
    combined = combined.reset_index(drop=True)

    logger.info(
        "Merged: %d base + %d OSM (capped %d/cat) = %d total places",
        len(base_df), len(osm_capped), max_osm_per_category, len(combined),
    )
    return combined
