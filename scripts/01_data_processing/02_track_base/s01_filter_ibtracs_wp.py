"""
IBTrACS - (WP)

：IBTrACSBASIN='WP'

：
- data/raw/IBTrACS/ibtracs.ALL.list.v04r01.csv

：
- reproduction_v6/02_processed_data/ibtracs/ibtracs_wp_filtered.csv

：
- BASIN = 'WP' ()
- SEASON >= 2000

Author: Typhoon-DA Reproduction
Date: 2026-02-02
"""

import pandas as pd
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPRO_ROOT = PROJECT_ROOT / "reproduction_v6"

INPUT_CSV = PROJECT_ROOT / "data" / "raw" / "IBTrACS" / "ibtracs.ALL.list.v04r01.csv"
OUTPUT_DIR = REPRO_ROOT / "02_processed_data" / "ibtracs"
OUTPUT_CSV = OUTPUT_DIR / "ibtracs_wp_filtered.csv"

def main():
    print("=" * 60)
    print("IBTrACS - (WP)")
    print("=" * 60)

    # 
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # （）
    print(f"\n[1] : {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, skiprows=[1], low_memory=False)
    print(f"    : {len(df):,}")

    # (BASIN='WP')
    print(f"\n[2] : BASIN='WP', SEASON>=2000")
    df_wp = df[(df['BASIN'] == 'WP') & (df['SEASON'] >= 2000)].copy()
    print(f"    : {len(df_wp):,}")

    # 
    print(f"\n[3] :")
    year_counts = df_wp['SEASON'].value_counts().sort_index()
    print(f"    : {year_counts.index.min()} - {year_counts.index.max()}")
    print(f"    : {df_wp['SID'].nunique()}")

    # 
    print(f"\n[4] : {OUTPUT_CSV}")
    df_wp.to_csv(OUTPUT_CSV, index=False)
    print(f"    : {os.path.getsize(OUTPUT_CSV) / (1024*1024):.2f} MB")

    # 
    print(f"\n[5]  (5):")
    preview_cols = ['SID', 'SEASON', 'NAME', 'ISO_TIME', 'LAT', 'LON', 'WMO_WIND', 'WMO_PRES']
    print(df_wp[preview_cols].head().to_string())

    print("\n" + "=" * 60)
    print("!")
    print("=" * 60)

if __name__ == "__main__":
    main()
