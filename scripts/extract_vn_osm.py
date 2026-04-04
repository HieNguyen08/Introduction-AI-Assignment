"""
extract_vn_osm.py — One-time script to extract Vietnam tourist POIs from Asia OSM .pbf.

Run once to produce data/osm/vietnam_places.csv, then the main pipeline uses that file.

Usage:
    python scripts/extract_vn_osm.py

Requirements:
    pip install osmium

Input:
    data/osm/asia-260327.osm.pbf  (~15 GB — streams node by node, not loaded into RAM)

Output:
    data/osm/vietnam_places.csv   (named tourist POIs inside Vietnam bounding box)

Runtime: ~30–60 minutes for the full Asia file (streams sequentially, ~2M nodes/sec).
"""

import os
import sys
import time

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OSM_DIR  = os.path.join(BASE_DIR, "data", "osm")
PBF_PATH = os.path.join(OSM_DIR, "asia-260327.osm.pbf")
OUT_CSV  = os.path.join(OSM_DIR, "vietnam_places.csv")

# ---------------------------------------------------------------------------
# Vietnam bounding box  (min_lat, min_lon, max_lat, max_lon)
# ---------------------------------------------------------------------------
VN_BBOX = (8.18, 102.14, 23.39, 109.46)

# ---------------------------------------------------------------------------
# Province mapping: assign nearest province by centroid distance
# ---------------------------------------------------------------------------
VN_PROVINCE_CENTROIDS = {
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
    "Phu Yen":         (13.0882, 109.0929),
    "Dak Lak":         (12.7100, 108.2378),
    "Ba Ria Vung Tau": (10.5417, 107.2429),
    "Son La":          (21.3270, 103.9144),
    "Yen Bai":         (21.7051, 104.8753),
    "Tuyen Quang":     (21.8231, 105.2142),
    "Phu Tho":         (21.3989, 105.2283),
    "Vinh Phuc":       (21.3009, 105.5474),
    "Bac Ninh":        (21.1861, 106.0763),
    "Hung Yen":        (20.6464, 106.0511),
    "Ha Nam":          (20.5418, 105.9221),
    "Nam Dinh":        (20.4388, 106.1621),
    "Thai Binh":       (20.4500, 106.3400),
    "Hai Duong":       (20.9373, 106.3147),
    "Quang Tri":       (16.7500, 107.1854),
    "Gia Lai":         (13.9833, 108.0000),
    "Kon Tum":         (14.3498, 108.0004),
    "Dak Nong":        (12.0000, 107.6917),
    "Binh Phuoc":      (11.7511, 106.7234),
    "Tay Ninh":        (11.3351, 106.1099),
    "Binh Duong":      (11.3254, 106.4770),
    "Dong Nai":        (11.0686, 107.1676),
    "Long An":         (10.5362, 106.4133),
    "Tien Giang":      (10.4493, 106.3421),
    "Ben Tre":         (10.2434, 106.3756),
    "Dong Thap":       (10.4938, 105.6882),
    "An Giang":        (10.3899, 105.4353),
    "Vinh Long":       (10.2538, 105.9722),
    "Can Tho":         (10.0452, 105.7469),
    "Hau Giang":       (9.7579,  105.6413),
    "Soc Trang":       (9.6025,  105.9740),
    "Tra Vinh":        (9.9347,  106.3456),
    "Ca Mau":          (9.1527,  105.1961),
    "Bac Lieu":        (9.2940,  105.7216),
    "Kien Giang":      (9.8250,  105.1259),
}

# ---------------------------------------------------------------------------
# OSM tag → category mapping (priority order matters — first match wins)
# ---------------------------------------------------------------------------
TAG_CATEGORY_RULES = [
    # beach
    ({"natural": "beach"},              "beach"),
    ({"tourism": "beach_resort"},       "beach"),
    ({"leisure": "beach_resort"},       "beach"),
    ({"natural": "bay"},                "beach"),
    # adventure
    ({"natural": "cave_entrance"},      "adventure"),
    ({"natural": "peak"},               "adventure"),
    ({"sport": "climbing"},             "adventure"),
    # nature
    ({"natural": "waterfall"},          "nature"),
    ({"natural": "hot_spring"},         "nature"),
    ({"natural": "spring"},             "nature"),
    ({"tourism": "viewpoint"},          "nature"),
    ({"leisure": "nature_reserve"},     "nature"),
    ({"leisure": "park"},               "nature"),
    ({"boundary": "national_park"},     "nature"),
    # culture
    ({"tourism": "museum"},             "culture"),
    ({"tourism": "attraction"},         "culture"),
    ({"tourism": "gallery"},            "culture"),
    ({"tourism": "artwork"},            "culture"),
    ({"historic": "monument"},          "culture"),
    ({"historic": "memorial"},          "culture"),
    ({"historic": "archaeological_site"}, "culture"),
    ({"historic": "castle"},            "culture"),
    ({"historic": "ruins"},             "culture"),
    ({"historic": "temple"},            "culture"),
    ({"historic": "pagoda"},            "culture"),
    ({"amenity": "place_of_worship"},   "culture"),
    # entertainment
    ({"tourism": "theme_park"},         "entertainment"),
    ({"tourism": "aquarium"},           "entertainment"),
    ({"tourism": "zoo"},                "entertainment"),
    ({"leisure": "water_park"},         "entertainment"),
    ({"leisure": "amusement_arcade"},   "entertainment"),
]


def get_category(tags: dict) -> str | None:
    for rule_tags, category in TAG_CATEGORY_RULES:
        if all(tags.get(k) == v for k, v in rule_tags.items()):
            return category
    return None


def get_name(tags: dict) -> str:
    return tags.get("name:en") or tags.get("name") or ""


def nearest_province(lat: float, lon: float) -> str:
    best, best_dist = "Unknown", float("inf")
    for prov, (plat, plon) in VN_PROVINCE_CENTROIDS.items():
        d = (lat - plat) ** 2 + (lon - plon) ** 2
        if d < best_dist:
            best_dist = d
            best = prov
    return best


def in_vietnam(lat: float, lon: float) -> bool:
    return VN_BBOX[0] <= lat <= VN_BBOX[2] and VN_BBOX[1] <= lon <= VN_BBOX[3]


# ---------------------------------------------------------------------------
# OSM handler
# ---------------------------------------------------------------------------
try:
    import osmium
    HAS_OSMIUM = True
except ImportError:
    HAS_OSMIUM = False


class VietnamTouristHandler(osmium.SimpleHandler if HAS_OSMIUM else object):
    """Streams the Asia .pbf and collects Vietnam tourist POIs (nodes only)."""

    def __init__(self):
        if HAS_OSMIUM:
            super().__init__()
        self.places: list[dict] = []
        self._processed = 0
        self._t0 = time.time()

    def node(self, n):
        self._processed += 1
        if self._processed % 10_000_000 == 0:
            elapsed = time.time() - self._t0
            rate = self._processed / elapsed / 1_000_000
            print(f"  {self._processed/1_000_000:.0f}M nodes | "
                  f"{len(self.places)} places found | "
                  f"{rate:.1f}M nodes/s | {elapsed/60:.1f} min elapsed")

        # Fast bbox pre-filter (avoids dict(n.tags) for 99% of nodes)
        lat = n.location.lat
        lon = n.location.lon
        if not in_vietnam(lat, lon):
            return

        tags = dict(n.tags)
        if not any(k in tags for k in ("tourism", "historic", "natural",
                                        "leisure", "boundary", "sport",
                                        "amenity")):
            return

        name = get_name(tags)
        if not name:
            return

        category = get_category(tags)
        if not category:
            return

        province = nearest_province(lat, lon)

        self.places.append({
            "place_name":    name,
            "latitude":      round(lat, 6),
            "longitude":     round(lon, 6),
            "category":      category,
            "province":      province,
            "entry_fee_vnd": 0,
            "osm_id":        n.id,
            "source":        "osm",
        })


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def postprocess(places: list[dict]):
    """Deduplicate, clean names, sort."""
    import pandas as pd

    df = pd.DataFrame(places)
    if df.empty:
        return df

    # Normalise name: strip, title-case if all-caps
    df["place_name"] = df["place_name"].str.strip()
    mask_upper = df["place_name"].str.isupper()
    df.loc[mask_upper, "place_name"] = df.loc[mask_upper, "place_name"].str.title()

    # Remove very short names (likely tags artifacts)
    df = df[df["place_name"].str.len() >= 4]

    # Deduplicate: same name + close coordinates (within ~500m ≈ 0.005°)
    df = df.sort_values("place_name").reset_index(drop=True)
    keep = []
    seen: list[tuple] = []
    for _, row in df.iterrows():
        dup = False
        for sname, slat, slon in seen:
            if (row["place_name"] == sname and
                    abs(row["latitude"] - slat) < 0.005 and
                    abs(row["longitude"] - slon) < 0.005):
                dup = True
                break
        if not dup:
            keep.append(row)
            seen.append((row["place_name"], row["latitude"], row["longitude"]))

    df = pd.DataFrame(keep).reset_index(drop=True)

    # Sort by category then name
    cat_order = {"nature": 0, "culture": 1, "beach": 2,
                 "adventure": 3, "entertainment": 4}
    df["_cat_order"] = df["category"].map(cat_order).fillna(9)
    df = df.sort_values(["_cat_order", "place_name"]).drop(columns="_cat_order")
    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not HAS_OSMIUM:
        print("ERROR: osmium not installed. Run: pip install osmium")
        sys.exit(1)

    if not os.path.exists(PBF_PATH):
        print(f"ERROR: PBF file not found: {PBF_PATH}")
        sys.exit(1)

    os.makedirs(OSM_DIR, exist_ok=True)

    print(f"Extracting Vietnam tourist POIs from Asia OSM .pbf")
    print(f"Input : {PBF_PATH}")
    print(f"Output: {OUT_CSV}")
    print(f"This may take 30–60 minutes for the 15 GB file...\n")

    handler = VietnamTouristHandler()
    t_start = time.time()
    handler.apply_file(PBF_PATH)
    elapsed = time.time() - t_start

    print(f"\nDone: {handler._processed/1e6:.1f}M nodes processed in "
          f"{elapsed/60:.1f} min")
    print(f"Raw Vietnam POIs found: {len(handler.places)}")

    import pandas as pd
    df = postprocess(handler.places)
    print(f"After deduplication: {len(df)} unique places")

    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved → {OUT_CSV}")

    # Category breakdown
    print("\nCategory breakdown:")
    for cat, count in df["category"].value_counts().items():
        print(f"  {cat:15s}: {count:4d} ({count/len(df):.0%})")

    # Province breakdown (top 15)
    print("\nTop provinces:")
    for prov, count in df["province"].value_counts().head(15).items():
        print(f"  {prov:20s}: {count}")


if __name__ == "__main__":
    main()
