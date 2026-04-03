"""
IBTrACSNetCDF

：IBTrACSNetCDF4，

：
- reproduction_v6/02_processed_data/ibtracs/ibtracs_wp_filtered.csv

：
- reproduction_v6/03_intermediate_nc/typhoon_{year}_base.nc

：
- center_lat:  (degrees_north)
- center_lon:  (degrees_east)
- wind_speed:  (knots)
- central_pressure:  (hPa)
- time:  (hours since 1970-01-01)
- radius_max_wind:  (nautical_miles)

Author: Typhoon-DA Reproduction
Date: 2026-02-02
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from netCDF4 import Dataset, Group

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPRO_ROOT = PROJECT_ROOT / "reproduction_v6"

INPUT_CSV = REPRO_ROOT / "02_processed_data" / "ibtracs" / "ibtracs_wp_filtered.csv"
OUTPUT_DIR = REPRO_ROOT / "03_intermediate_nc"
LOG_FILE = REPRO_ROOT / "06_logs" / "01_ibtracs_to_nc.log"

# 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class IBTrACSNetCDFConverter:
    """IBTrACS CSVNetCDF"""

    FILL_VALUE = -999.0

    VALID_RANGES = {
        'latitude': (-90, 90),
        'longitude': (-180, 180),
        'wind_speed': (0, 300),
        'pressure': (850, 1100),
    }

    def __init__(self, input_csv: str, output_dir: str):
        self.input_csv = Path(input_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data = None
        self.stats = {
            'total_records': 0,
            'typhoon_count': 0,
            'years_processed': [],
            'files_created': []
        }

    def load_data(self) -> pd.DataFrame:
        """CSV"""
        logger.info(f": {self.input_csv}")
        df = pd.read_csv(self.input_csv, low_memory=False)

        # 
        df['datetime'] = pd.to_datetime(df['ISO_TIME'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

        # 2000-2024
        df = df[(df['SEASON'] >= 2000) & (df['SEASON'] <= 2024)]

        self.data = df
        self.stats['total_records'] = len(df)
        self.stats['typhoon_count'] = df['SID'].nunique()
        logger.info(f"  : {len(df):,}")
        logger.info(f"  : {df['SID'].nunique()}")
        return df

    def convert_to_timestamp(self, dt_series: pd.Series) -> np.ndarray:
        """datetime1970-01-01"""
        epoch = pd.Timestamp('1970-01-01 00:00:00')
        hours = (dt_series - epoch).dt.total_seconds() / 3600.0
        return hours.values

    def replace_missing(self, series: pd.Series) -> np.ndarray:
        """FILL_VALUE"""
        numeric = pd.to_numeric(series, errors='coerce')
        arr = numeric.values.copy()
        arr[np.isnan(arr)] = self.FILL_VALUE
        return arr

    def create_netcdf_for_year(self, year: int, year_data: pd.DataFrame) -> str:
        """NetCDF"""
        output_file = self.output_dir / f"typhoon_{year}_base.nc"
        logger.info(f"NetCDF: {output_file.name}")

        with Dataset(output_file, 'w', format='NETCDF4') as nc:
            # 
            nc.title = "Typhoon Track Data - IBTrACS"
            nc.source = "IBTrACS v04r01"
            nc.Conventions = "CF-1.8"
            nc.creation_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            nc.year = year

            # 
            typhoon_groups = year_data.groupby('SID')
            nc.num_typhoons = len(typhoon_groups)

            for sid, group in typhoon_groups:
                group = group.sort_values('datetime')
                typhoon_name = group['NAME'].iloc[0]
                typhoon_number = group['NUMBER'].iloc[0]

                # Group
                group_name = f"{year}_{typhoon_name}_{typhoon_number}".replace(' ', '_')
                if group_name in nc.groups:
                    group_name = f"{group_name}_{sid}"

                grp = nc.createGroup(group_name)

                # 
                n_time = len(group)
                grp.createDimension('time', n_time)

                # 
                time_var = grp.createVariable('time', 'f8', ('time',), zlib=True)
                time_var.units = 'hours since 1970-01-01 00:00:00'
                time_var.standard_name = 'time'
                time_var[:] = self.convert_to_timestamp(group['datetime'])

                # 
                lon_var = grp.createVariable('center_lon', 'f4', ('time',),
                                             zlib=True, fill_value=self.FILL_VALUE)
                lon_var.units = 'degrees_east'
                lon_var.standard_name = 'longitude'
                lon_var.long_name = 'Typhoon center longitude'
                lon_var[:] = self.replace_missing(group['LON'])

                # 
                lat_var = grp.createVariable('center_lat', 'f4', ('time',),
                                             zlib=True, fill_value=self.FILL_VALUE)
                lat_var.units = 'degrees_north'
                lat_var.standard_name = 'latitude'
                lat_var.long_name = 'Typhoon center latitude'
                lat_var[:] = self.replace_missing(group['LAT'])

                # 
                wind_var = grp.createVariable('wind_speed', 'f4', ('time',),
                                              zlib=True, fill_value=self.FILL_VALUE)
                wind_var.units = 'knots'
                wind_var.standard_name = 'wind_speed'
                wind_var.long_name = 'Maximum sustained wind speed'
                wind_var[:] = self.replace_missing(group['WMO_WIND'])

                # 
                pres_var = grp.createVariable('central_pressure', 'f4', ('time',),
                                              zlib=True, fill_value=self.FILL_VALUE)
                pres_var.units = 'hPa'
                pres_var.standard_name = 'air_pressure'
                pres_var.long_name = 'Central pressure'
                pres_var[:] = self.replace_missing(group['WMO_PRES'])

                # 
                rmw_var = grp.createVariable('radius_max_wind', 'f4', ('time',),
                                             zlib=True, fill_value=self.FILL_VALUE)
                rmw_var.units = 'nautical_miles'
                rmw_var.long_name = 'Radius of maximum wind speed'
                rmw_var[:] = self.replace_missing(group['USA_RMW'])

                # Group
                grp.typhoon_id = sid
                grp.typhoon_name = typhoon_name
                grp.season = int(year)
                grp.number = int(typhoon_number)
                grp.start_time = group['ISO_TIME'].iloc[0]
                grp.end_time = group['ISO_TIME'].iloc[-1]

        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        logger.info(f"  : {file_size_mb:.2f} MB, {n_time} ")
        self.stats['files_created'].append(str(output_file))
        self.stats['years_processed'].append(year)

        return str(output_file)

    def process_all_years(self) -> List[str]:
        """"""
        logger.info("=" * 60)
        logger.info("IBTrACSNetCDF")
        logger.info("=" * 60)

        data = self.load_data()
        years = sorted(data['SEASON'].unique())

        logger.info(f" {len(years)} : {years[0]} - {years[-1]}")
        logger.info("-" * 60)

        output_files = []
        for year in years:
            year_data = data[data['SEASON'] == year]
            output_file = self.create_netcdf_for_year(year, year_data)
            output_files.append(output_file)

        logger.info("=" * 60)
        logger.info(f"!  {len(output_files)} NetCDF")
        logger.info("=" * 60)

        return output_files


def main():
    print("=" * 60)
    print("IBTrACSNetCDF")
    print("=" * 60)

    converter = IBTrACSNetCDFConverter(INPUT_CSV, OUTPUT_DIR)
    output_files = converter.process_all_years()

    # 
    print("\n:")
    for f in output_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  {os.path.basename(f)}: {size_mb:.2f} MB")

    print(f"\n: {LOG_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
