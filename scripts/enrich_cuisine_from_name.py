"""
enrich_cuisine_from_name.py — Infer cuisine type from restaurant name keywords.

Targets: vn_restaurants.csv rows where cuisine == 'unknown' (77.4% of 1521 rows).
Uses Vietnamese + English name patterns to classify into 10 cuisine buckets.

Usage:
    python scripts/enrich_cuisine_from_name.py

Output:
    data/features/vn_restaurants.csv  (updated in-place)
    Prints cuisine coverage before/after.
"""

import os, re, sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(BASE_DIR, "data", "features", "vn_restaurants.csv")

# ─── Keyword → cuisine bucket ─────────────────────────────────────────────────
# Vietnamese keywords (both ASCII and Unicode)
PATTERNS: list[tuple[list[str], str]] = [
    # Cafe / Drinks — MUST be before vietnamese to catch "Cà Phê" first
    (["cà phê", "ca phe", "café", "cafe", "coffee", "trà sữa", "tra sua",
      "bubble tea", "trà chanh", "juice", "sinh tố", "smoothie",
      "kem", "ice cream", "bakery", "bánh ngọt", "dessert", "bánh mì cafe"], "cafe"),

    # Vietnamese
    (["phở", "pho", "bún", "bun", "bánh mì", "banh mi", "cơm", "com",
      "xôi", "xoi", "chả", "cha", "nầm", "lẩu", "bò", "heo", "gà",
      "nhà hàng việt", "viet", "com tam", "cơm tấm", "bún bò",
      "bánh cuốn", "bánh xèo", "bánh tráng", "bún riêu", "cháo",
      "miến", "mì quảng", "hủ tiếu", "hu tieu", "nhà hàng chay",
      "chay", "vegetarian vietnamese", "vietnamese"], "vietnamese"),

    # Seafood
    (["hải sản", "hai san", "seafood", "tôm", "cua", "mực", "cá",
      "oyster", "sò", "nghêu", "lobster", "shrimp"], "seafood"),

    # Japanese
    (["sushi", "ramen", "nhật", "japan", "hokkaido", "sakura",
      "tokyo", "osaka", "tempura", "sashimi", "udon", "izakaya",
      "wasabi", "matcha restaurant"], "japanese"),

    # Korean
    (["hàn", "han quoc", "korea", "korean", "bbq hàn", "lẩu hàn",
      "topokki", "ramyeon", "kimbap", "bibimbap", "samgyeopsal"], "korean"),

    # Chinese
    (["trung hoa", "china", "chinese", "dim sum", "canton", "lẩu trung",
      "mì trung hoa", "hong kong", "beijing", "shanghai restaurant",
      "yum cha"], "chinese"),

    # Asian (mixed / Thai / Indian / other Asian)
    (["asian", "thái lan", "thailand", "thai", "ấn độ", "india", "indian",
      "tandoor", "khazaana", "curry", "pan asian", "indochine",
      "tamarind", "indonesian", "singaporean", "malaysia"], "asian"),

    # BBQ / Hotpot
    (["nướng", "nuong", "bbq", "grill", "lẩu", "lau", "hotpot",
      "yakiniku", "barbecue", "buffet nướng"], "bbq"),

    # Western / European
    (["pizza", "pasta", "italian", "pháp", "french", "steak",
      "burger", "beefsteak", "beefsteakhouse", "mediterranean",
      "greek", "spanish", "european", "american restaurant",
      "brazil", "grill house", "brasserie", "bistro",
      "classico", "mediterraneo", "la badiane", "little italian",
      "hoa sữa", "gecko", "ladybird", "new day"], "western"),

    # Fast food / Sandwiches
    (["kfc", "mcdonald", "burger king", "jollibee", "lotteria",
      "popeyes", "subway", "fast food", "fried chicken",
      "sandwich shop", "bánh mì shop"], "fast_food"),

    # Pub / Bar — keep as separate bucket
    (["pub", "bar ", "tavern", "irish", "beer", "bia", "wolfhound",
      "finnegan", "mika pub", "r&r"], "cafe"),   # treat as cafe bucket
]


def infer_cuisine_from_name(name: str) -> str:
    """Match name (Vietnamese + English) against cuisine keyword patterns."""
    name_lower = name.lower()
    for keywords, cuisine in PATTERNS:
        for kw in keywords:
            if kw in name_lower:
                return cuisine
    return "unknown"


def main():
    if not os.path.exists(OUT_PATH):
        print(f"ERROR: {OUT_PATH} not found. Run run_full_pipeline() first.")
        sys.exit(1)

    df = pd.read_csv(OUT_PATH)
    before = (df['cuisine'] != 'unknown').sum()
    total = len(df)
    print(f"Restaurants: {total} rows")
    print(f"Cuisine known BEFORE: {before}/{total} ({before/total*100:.1f}%)")

    # Only update rows where cuisine == 'unknown'
    mask = df['cuisine'] == 'unknown'
    df.loc[mask, 'cuisine'] = df.loc[mask, 'name'].apply(infer_cuisine_from_name)

    after = (df['cuisine'] != 'unknown').sum()
    print(f"Cuisine known AFTER:  {after}/{total} ({after/total*100:.1f}%)")
    print(f"Newly classified:     {after - before} rows")

    print(f"\nCuisine distribution after enrichment:")
    for cuisine, count in df['cuisine'].value_counts().items():
        print(f"  {cuisine:20s}: {count:4d} ({count/total*100:.1f}%)")

    # Normalize remaining raw OSM tags that weren't caught above
    RAW_TAG_MAP = {
        "international": "western", "fusion": "western",
        "steak_house": "western", "steakhouse": "western",
        "argentinian": "western", "australian": "western",
        "danish": "western", "german": "western",
        "spanish": "western", "lebanese": "asian",
        "middle_eastern": "asian", "mexican": "western",
        "italian_pizza": "western",
        "milk_tea": "cafe", "tea_leisure": "cafe",
        "japan_food": "japanese", "beef_bowl": "japanese",
        "nướng": "bbq", "lau": "bbq",
    }
    # Normalize any multi-word / compound raw values
    def normalize_raw(c: str) -> str:
        c_low = c.lower().replace(" ", "_")
        if c_low in RAW_TAG_MAP:
            return RAW_TAG_MAP[c_low]
        # Partial match
        for raw, mapped in RAW_TAG_MAP.items():
            if raw in c_low:
                return mapped
        # Vietnamese multi-word that contains a known bucket
        for bucket in ["vietnamese", "seafood", "japanese", "korean", "chinese",
                       "asian", "bbq", "western", "fast_food", "cafe"]:
            if bucket in c_low:
                return bucket
        return c  # keep as-is if still unmatched

    valid_buckets = {"vietnamese", "seafood", "japanese", "korean", "chinese",
                     "asian", "bbq", "western", "fast_food", "cafe", "unknown"}
    df['cuisine'] = df['cuisine'].apply(
        lambda c: c if c in valid_buckets else normalize_raw(c)
    )
    # Any still not in buckets → unknown
    df['cuisine'] = df['cuisine'].apply(
        lambda c: c if c in valid_buckets else "unknown"
    )

    final_known = (df['cuisine'] != 'unknown').sum()
    print(f"\nAfter normalization: {final_known}/{total} ({final_known/total*100:.1f}%) known")
    print(f"\nFinal cuisine distribution:")
    for cuisine, count in df['cuisine'].value_counts().items():
        print(f"  {cuisine:20s}: {count:4d} ({count/total*100:.1f}%)")

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
