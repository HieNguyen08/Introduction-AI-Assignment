"""
scrape_vn_weather_extra.py — Scrape supplemental weather data for under-represented provinces.

EDA finding: weather_probabilities.csv covers 40 provinces but some have sparse historical data.
This script scrapes monthly climate summaries from climatedata.org (public, no login).

Usage:
    python scripts/scrape_vn_weather_extra.py

Output:
    data/scraped/vn_climate_monthly.csv  — province, month, avg_temp, rain_days, humidity

Requirements:
    pip install "scrapling[fetchers]"
"""

import os
import re
import time
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE_DIR, "data", "scraped")
os.makedirs(OUT_DIR, exist_ok=True)

# climatedata.org slugs for key VN cities
CLIMATE_TARGETS = [
    ("Ha Noi",      "https://en.climate-data.org/asia/vietnam/ha-noi/hanoi-714/"),
    ("Ho Chi Minh", "https://en.climate-data.org/asia/vietnam/ho-chi-minh-city/ho-chi-minh-city-2/"),
    ("Da Nang",     "https://en.climate-data.org/asia/vietnam/da-nang/danang-1209/"),
    ("Hue",         "https://en.climate-data.org/asia/vietnam/thua-thien-hue/hue-714428/"),
    ("Nha Trang",   "https://en.climate-data.org/asia/vietnam/khanh-hoa/nha-trang-715/"),
    ("Da Lat",      "https://en.climate-data.org/asia/vietnam/lam-dong/da-lat-716/"),
    ("Ha Long",     "https://en.climate-data.org/asia/vietnam/quang-ninh/ha-long-717/"),
    ("Hoi An",      "https://en.climate-data.org/asia/vietnam/quang-nam/hoi-an-62527/"),
    ("Can Tho",     "https://en.climate-data.org/asia/vietnam/can-tho/cantho-718/"),
    ("Phu Quoc",    "https://en.climate-data.org/asia/vietnam/kien-giang/phu-quoc-719/"),
    ("Sapa",        "https://en.climate-data.org/asia/vietnam/lao-cai/sapa-720/"),
    ("Ninh Binh",   "https://en.climate-data.org/asia/vietnam/ninh-binh/ninh-binh-722/"),
    ("Mui Ne",      "https://en.climate-data.org/asia/vietnam/binh-thuan/mui-ne-723/"),
    ("Ha Giang",    "https://en.climate-data.org/asia/vietnam/ha-giang/ha-giang-724/"),
]

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


def scrape_climate_page(province: str, url: str) -> list[dict]:
    """Scrape monthly climate table from climatedata.org."""
    from scrapling.fetchers import Fetcher

    records = []
    try:
        page = Fetcher.get(url, impersonate="chrome", timeout=25)
        if page.status != 200:
            log.warning(f"  {province}: HTTP {page.status}")
            return records

        # climatedata.org: Table index 1 has months as columns, metrics as rows
        # Data is inside nested child elements — must use *::text
        tables = page.css("table")
        if len(tables) < 2:
            log.warning(f"  {province}: expected ≥2 tables, found {len(tables)}")
            return records

        table = tables[1]  # month-as-columns layout
        rows = table.css("tr")

        def extract_row(row) -> tuple[str, list[str]]:
            """Return (label, [12 values]) from a tr using *::text for nested elements."""
            cells = row.css("td, th")
            if not cells:
                return "", []
            label = " ".join(
                t.strip() for t in cells[0].css("*::text").getall() if t.strip()
            ).lower()
            values = []
            for cell in cells[1:]:
                # Take first numeric-looking text in each cell
                texts = [t.strip() for t in cell.css("*::text").getall() if t.strip()]
                # Prefer °C values; fall back to first token
                num_text = next(
                    (t for t in texts if re.search(r"[\-\d]", t) and "°f" not in t.lower()),
                    texts[0] if texts else ""
                )
                values.append(num_text)
            return label, values

        row_data: dict[str, list] = {}
        for row in rows:
            label, values = extract_row(row)
            if label and values:
                row_data[label] = values

        def parse_nums(vals: list[str]) -> list[float | None]:
            result = []
            for v in vals[:12]:
                # Strip °C, %, spaces then parse first float
                v_clean = re.sub(r"[°c%\s]", "", v, flags=re.IGNORECASE)
                m = re.search(r"[\-\d\.]+", v_clean)
                result.append(float(m.group()) if m else None)
            while len(result) < 12:
                result.append(None)
            return result

        # Match row labels from climatedata.org
        temp_key  = next((k for k in row_data if "avg" in k and "temp" in k), None)
        rain_key  = next((k for k in row_data if "precipitation" in k or ("rain" in k and "day" not in k)), None)
        humid_key = next((k for k in row_data if "humid" in k), None)
        days_key  = next((k for k in row_data if "rainy day" in k or ("day" in k and "rain" in k)), None)

        temps     = parse_nums(row_data.get(temp_key, [])) if temp_key else [None]*12
        rains     = parse_nums(row_data.get(rain_key, [])) if rain_key else [None]*12
        humids    = parse_nums(row_data.get(humid_key, [])) if humid_key else [None]*12
        rain_days = parse_nums(row_data.get(days_key, [])) if days_key else [None]*12

        for i, month_name in enumerate(MONTHS):
            records.append({
                "province":      province,
                "month":         i + 1,
                "month_name":    month_name,
                "avg_temp_c":    temps[i],
                "rain_mm":       rains[i],
                "humidity_pct":  humids[i],
                "rain_days":     rain_days[i],
                "source":        "climate-data.org",
            })

        log.info(f"  {province}: {len([r for r in records if r['avg_temp_c'] is not None])} months with temp data")

    except Exception as e:
        log.error(f"  {province}: {e}")

    return records


def main():
    all_records = []

    for province, url in CLIMATE_TARGETS:
        records = scrape_climate_page(province, url)
        all_records.extend(records)
        time.sleep(1.2)

    if not all_records:
        log.error("No climate data scraped.")
        return

    df = pd.DataFrame(all_records)
    out_path = os.path.join(OUT_DIR, "vn_climate_monthly.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    log.info(f"Saved → {out_path} ({len(df)} rows)")

    # Summary
    print(f"\nClimate records: {len(df)} ({df['province'].nunique()} provinces × 12 months)")
    complete = df.dropna(subset=["avg_temp_c"])
    print(f"Complete temp data: {len(complete)}/{len(df)} ({len(complete)/len(df)*100:.1f}%)")

    if len(complete) > 0:
        print(f"\nSample (Ha Noi):")
        ha_noi = df[df["province"] == "Ha Noi"][["month_name","avg_temp_c","rain_mm","humidity_pct"]]
        if not ha_noi.empty:
            print(ha_noi.to_string(index=False))


if __name__ == "__main__":
    main()
