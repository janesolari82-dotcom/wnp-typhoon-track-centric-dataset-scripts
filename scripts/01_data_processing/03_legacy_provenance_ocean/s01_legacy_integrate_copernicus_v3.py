"""
 - 4D ()

：CopernicusNetCDF
：15×41×41

Author: Typhoon-DA Reproduction
Date: 2026-02-02
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from netCDF4 import Dataset, Group

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPRO_ROOT = PROJECT_ROOT / "reproduction_v6"

BASE_NC_DIR = REPRO_ROOT / "03_intermediate_nc"
COPERNICUS_DIR = PROJECT_ROOT / "data" / "raw" / "Copernicus_GlobalOcean_Physics_2000-2024"
OUTPUT_DIR = REPRO_ROOT / "03_intermediate_nc" / "ocean_integration"
LOG_FILE = REPRO_ROOT / "06_logs" / "02_copernicus_integration.log"

BUFFER_DAYS = 7
WINDOW_KM = 370.0
N_WINDOW_LAT = 41
N_WINDOW_LON = 41
N_WINDOW_TIME = BUFFER_DAYS * 2 + 1  # 15
FILL_VALUE = -9999.0

OCEAN_VARS = {
    'ocean_thetao': {'source': 'thetao', 'units': 'degC', 'long_name': 'Sea surface temperature'},
    'ocean_zos': {'source': 'zos', 'units': 'm', 'long_name': 'Sea surface height anomaly'},
    'ocean_so': {'source': 'so', 'units': 'PSU', 'long_name': 'Sea surface salinity'},
    'ocean_uo': {'source': 'uo', 'units': 'm/s', 'long_name': 'Eastward sea water velocity'},
    'ocean_vo': {'source': 'vo', 'units': 'm/s', 'long_name': 'Northward sea water velocity'},
    'ocean_mlotst': {'source': 'mlotst', 'units': 'm', 'long_name': 'Mixed layer thickness'},
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class OceanIntegrator:
    def __init__(self, base_nc_dir: str, copernicus_dir: str, output_dir: str):
        self.base_nc_dir = Path(base_nc_dir)
        self.copernicus_dir = Path(copernicus_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.copernicus_data: Dict[int, Dataset] = {}

        self.window_deg = WINDOW_KM / 2.0 / 111.0
        self.window_lat_offsets = np.linspace(-self.window_deg, self.window_deg, N_WINDOW_LAT, dtype=np.float32)
        self.window_lon_offsets = np.linspace(-self.window_deg, self.window_deg, N_WINDOW_LON, dtype=np.float32)
        self.window_time_offsets = np.arange(-BUFFER_DAYS, BUFFER_DAYS + 1, dtype=np.float32)

        # ：1970-01-01  1950-01-01
        self.hours_1970_to_1950 = 20 * 365.25 * 24

        self.stats = {
            'years': [],
            'typhoons': 0,
            'success': 0,
            'fail': 0,
        }

        logger.info("=" * 60)
        logger.info("Ocean Data Integration (4D Sliding Window)")
        logger.info("=" * 60)

    def load_copernicus(self, year: int) -> Optional[Dataset]:
        if year in self.copernicus_data:
            return self.copernicus_data[year]

        year_dir = self.copernicus_dir / str(year)
        if year_dir.exists():
            files = list(year_dir.glob("*.nc"))
            if files:
                ds = Dataset(str(files[0]), 'r')
                self.copernicus_data[year] = ds
                return ds
        return None

    def get_ocean_4d_window(self, year: int, lat: float, lon: float,
                            typhoon_time_hours: float) -> Dict[str, np.ndarray]:
        """4D (window_time, window_lat, window_lon)"""
        cop_ds = self.load_copernicus(year)
        if cop_ds is None:
            return {}

        ocean_lats = cop_ds.variables['latitude'][:]
        ocean_lons = cop_ds.variables['longitude'][:]
        ocean_times = cop_ds.variables['time'][:]

        # 
        ocean_time_hours = typhoon_time_hours + self.hours_1970_to_1950

        # 
        lat_start = np.searchsorted(ocean_lats, lat - self.window_deg)
        lat_end = np.searchsorted(ocean_lats, lat + self.window_deg)
        lon_start = np.searchsorted(ocean_lons, lon - self.window_deg)
        lon_end = np.searchsorted(ocean_lons, lon + self.window_deg)

        lat_start = max(0, lat_start)
        lat_end = min(len(ocean_lats), lat_end)
        lon_start = max(0, lon_start)
        lon_end = min(len(ocean_lons), lon_end)

        # 
        t_start = np.searchsorted(ocean_times, ocean_time_hours - BUFFER_DAYS * 24)
        t_end = np.searchsorted(ocean_times, ocean_time_hours + BUFFER_DAYS * 24)
        t_start = max(0, t_start)
        t_end = min(len(ocean_times), t_end)

        result = {}

        for var_name, info in OCEAN_VARS.items():
            src = info['source']
            if src not in cop_ds.variables:
                continue

            try:
                var = cop_ds.variables[src]
                if src == 'mlotst':
                    data = var[t_start:t_end, lat_start:lat_end, lon_start:lon_end]
                else:
                    data = var[t_start:t_end, 0, lat_start:lat_end, lon_start:lon_end]

                # 
                result[var_name] = self._pad_to_window(data.astype(np.float32))
            except Exception as e:
                pass

        return result

    def _pad_to_window(self, data: np.ndarray) -> np.ndarray:
        """"""
        shape = data.shape
        if len(shape) == 3:
            _, h, w = shape
            out = np.full((N_WINDOW_TIME, N_WINDOW_LAT, N_WINDOW_LON),
                         FILL_VALUE, dtype=np.float32)
            out[:min(shape[0], N_WINDOW_TIME),
                :min(h, N_WINDOW_LAT),
                :min(w, N_WINDOW_LON)] = data[:min(shape[0], N_WINDOW_TIME),
                                              :min(h, N_WINDOW_LAT),
                                              :min(w, N_WINDOW_LON)]
            return out
        return np.full((N_WINDOW_TIME, N_WINDOW_LAT, N_WINDOW_LON),
                      FILL_VALUE, dtype=np.float32)

    def process_year(self, year: int) -> str:
        inp = self.base_nc_dir / f"typhoon_{year}_base.nc"
        out = self.output_dir / f"typhoon_{year}_ocean.nc"

        if not inp.exists():
            return ""

        logger.info(f"\n[{year}] Processing...")

        cop_ds = self.load_copernicus(year)
        if cop_ds is None:
            logger.warning(f"  No Copernicus data for {year}")
            return ""

        with Dataset(inp, 'r') as src:
            with Dataset(out, 'w', format='NETCDF4') as dst:
                # 
                dst.title = src.title
                dst.Conventions = "CF-1.8"
                dst.creation_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                dst.year = year

                typhoon_cnt = 0

                for gname, sgrp in src.groups.items():
                    dgrp = dst.createGroup(gname)

                    lats = sgrp.variables['center_lat'][:]
                    lons = sgrp.variables['center_lon'][:]
                    times = sgrp.variables['time'][:]
                    n = len(lats)

                    # 
                    dgrp.createDimension('time', n)
                    dgrp.createDimension('window_time', N_WINDOW_TIME)
                    dgrp.createDimension('window_lat', N_WINDOW_LAT)
                    dgrp.createDimension('window_lon', N_WINDOW_LON)

                    # 
                    wt = dgrp.createVariable('window_time', 'f4', ('window_time',))
                    wt.units = 'days'
                    wt[:] = self.window_time_offsets

                    wl = dgrp.createVariable('window_lat', 'f4', ('window_lat',))
                    wl.units = 'degrees'
                    wl[:] = self.window_lat_offsets

                    wlo = dgrp.createVariable('window_lon', 'f4', ('window_lon',))
                    wlo.units = 'degrees'
                    wlo[:] = self.window_lon_offsets

                    # 
                    for vn in ['time', 'center_lat', 'center_lon', 'wind_speed',
                               'central_pressure', 'radius_max_wind']:
                        if vn in sgrp.variables:
                            sv = sgrp.variables[vn]
                            dv = dgrp.createVariable(vn, sv.dtype, sv.dimensions,
                                                    zlib=True, fill_value=FILL_VALUE)
                            for attr in sv.ncattrs():
                                if attr not in ['_FillValue', 'missing_value']:
                                    try:
                                        dv.setncattr(attr, sv.getncattr(attr))
                                    except:
                                        pass
                            dv[:] = sv[:]

                    # 
                    ocean_vars = {}
                    for vn, info in OCEAN_VARS.items():
                        dv = dgrp.createVariable(vn, 'f4',
                                                  ('time', 'window_time', 'window_lat', 'window_lon'),
                                                  zlib=True, fill_value=FILL_VALUE)
                        dv.units = info['units']
                        dv.long_name = info['long_name']
                        dv.missing_value = FILL_VALUE
                        ocean_vars[vn] = dv

                    # 
                    success = 0
                    for t in range(n):
                        lat, lon = float(lats[t]), float(lons[t])
                        if lat < -90 or lat > 90 or lon < -180 or lon > 180:
                            continue

                        data = self.get_ocean_4d_window(year, lat, lon, float(times[t]))

                        if data:
                            for vn, d in data.items():
                                ocean_vars[vn][t] = d
                            success += 1
                            self.stats['success'] += 1
                        else:
                            self.stats['fail'] += 1

                    self.stats['typhoons'] += 1
                    if success > 0:
                        typhoon_cnt += 1
                        logger.info(f"    {gname}: {success}/{n}")

                dst.num_typhoons = typhoon_cnt
                self.stats['years'].append(year)

        logger.info(f"  Done: {out.name}")
        return str(out)

    def process_all(self, start=2000, end=2024):
        logger.info("=" * 60)
        logger.info("Starting Ocean Data Integration")
        logger.info("=" * 60)

        outputs = []
        for year in range(start, end + 1):
            out = self.process_year(year)
            if out:
                outputs.append(out)

        logger.info("\n" + "=" * 60)
        logger.info("Complete!")
        logger.info(f"Years: {len(self.stats['years'])}")
        logger.info(f"Typhoons: {self.stats['typhoons']}")
        logger.info(f"Success: {self.stats['success']}, Fail: {self.stats['fail']}")
        logger.info("=" * 60)

        return outputs

    def close(self):
        for ds in self.copernicus_data.values():
            ds.close()
        self.copernicus_data.clear()


def main():
    print("=" * 60)
    print("Ocean Data Integration (4D Sliding Window)")
    print("=" * 60)

    integrator = OceanIntegrator(BASE_NC_DIR, COPERNICUS_DIR, OUTPUT_DIR)
    outputs = integrator.process_all()
    integrator.close()

    print("\nOutput files:")
    for f in outputs:
        print(f"  {os.path.basename(f)}: {os.path.getsize(f)/(1024*1024):.1f} MB")

    print(f"\nLog: {LOG_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
