from dan_weather_suite.models.loader import ModelLoader
import dan_weather_suite.plotting.regions as regions
import dan_weather_suite.utils as utils
from datetime import datetime, time, timedelta
from numpy.typing import NDArray
import logging
import numpy as np
import os
import pickle
from scipy.spatial import KDTree
from typing import Tuple
import time as ttime
import xarray as xr


logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


class NbmLoader(ModelLoader):
    def __init__(self, short_term=False):
        super().__init__()

        self.hourly = list(range(1, 36 + 1, 1))
        self.six_hourly = list(range(42, 240 + 6, 6))
        self.forecast_hours = self.hourly + self.six_hourly
        self.short_term_length = 60

        if short_term:
            self.forecast_hours = [
                t for t in self.forecast_hours if t <= self.short_term_length
            ]
            self.grib_file = "grib/nbm-short.grib"
            self.netcdf_file = "grib/nbm-short.nc"

        else:
            self.grib_file = "grib/nbm.grib"
            self.netcdf_file = "grib/nbm.nc"


    def get_latest_init(self) -> datetime:
        current_utc = datetime.utcnow()
        current_utc_time = current_utc.time()

        release_00z = time(1, 15)
        release_12z = time(13, 15)

        if release_00z <= current_utc_time < release_12z:
            return datetime(current_utc.year, current_utc.month, current_utc.day, 0, 0)
        elif current_utc_time >= release_12z:
            return datetime(current_utc.year, current_utc.month, current_utc.day, 12, 0)
        else:
            previous_day = current_utc - timedelta(days=1)
            return datetime(
                previous_day.year, previous_day.month, previous_day.day, 12, 0
            )

    def url_formatter(self, init_dt: datetime, fhour: int) -> Tuple[str, dict]:
        day_str = init_dt.strftime("%Y%m%d")
        cycle_str = str(init_dt.hour).zfill(2)
        fhour_str = str(fhour).zfill(3)

        extent = regions.PRISM_EXTENT
        left = extent.left
        right = extent.right
        top = extent.top
        bottom = extent.bottom

        base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_blend.pl"

        params = {
            "dir": f"/blend.{day_str}/{cycle_str}/core",
            "file": f"blend.t{cycle_str}z.core.f{fhour_str}.co.grib2",
            "var_SNOWLR": "on",
            "subregion": "",
            "toplat": top,
            "leftlon": left,
            "rightlon": right,
            "bottomlat": bottom,
        }

        return base_url, params

    def download_grib(self, cycle=None):
        init_dt = self.get_latest_init()
        if cycle is not None:
            init_dt = init_dt.replace(hour=cycle)

        # Need to download serially...slowly because of NOAA grib server request limits

        if os.path.exists(self.grib_file):
            os.remove(self.grib_file)

        with open(self.grib_file, "ab") as f:
            for fhour in self.forecast_hours:
                url, params = self.url_formatter(
                    init_dt,
                    fhour,
                )
                data = utils.download_bytes(url, params)
                if data:
                    f.write(data)
                ttime.sleep(0.1)

    def extrapolate_to_init_time(self, ds: xr.Dataset) -> xr.Dataset:
        "NBM does not come with init_time data, extrapolate to init time"
        init_timedelta = np.timedelta64(0, "ns")
        if ds.step[0] != init_timedelta:
            steps = np.concatenate([[init_timedelta], ds.step.values])
            ds = ds.interp(step=steps, kwargs={"fill_value": "extrapolate"})
            new_times = ds.time + ds.step
            ds = ds.assign_coords(valid_time=new_times)
        return ds

    def process_grib(self) -> xr.Dataset:
        ds = xr.open_dataset(self.grib_file, chunks={"step": 1})
        ds = ds.rename({"unknown": "slr"})
        ds["longitude"] = (ds["longitude"] + 180) % 360 - 180
        ds["slr"] = ds.slr.fillna(0).clip(0, None)
        ds = self.extrapolate_to_init_time(ds)
        ds = ds.swap_dims({"step": "valid_time"})
        ds = ds.chunk({"valid_time": 1})
        return ds


    def create_kdtree(self) -> KDTree:
        "Create KD tree of ds coordinates"
        ds = self.open_dataset()
        pairs = np.dstack((ds.longitude, ds.latitude)).reshape(-1, 2)
        tree = KDTree(pairs)
        with open("nbm-tree.pkl", "wb") as f:
            pickle.dump(tree, f)

        return tree

    def forecast_slr(self, lon, lat) -> NDArray:
        ds = self.open_dataset()
        k_nearest = 36
        if not os.path.exists("nbm-tree.pkl"):
            tree = self.create_kdtree()
        else:
            with open("nbm-tree.pkl", "rb") as f:
                tree = pickle.load(f)

        pairs = np.dstack((ds.longitude, ds.latitude)).reshape(-1, 2)
        distances, indicies = tree.query([lon, lat], k_nearest)
        nearest_points = pairs[indicies]

        # (haversine dist, coord array indices)
        nearest_points_idx = [
            (utils.haversine(lat, lon, nbm_lat, nbm_lon), idx)
            for (nbm_lon, nbm_lat), idx in zip(nearest_points, indicies)
        ]

        nearest_point_km, nearest_idx = min(nearest_points_idx, key=lambda x: x[0])
        logging.info(f"Nearest NBM: {pairs[nearest_idx]} {round(nearest_point_km,2)}km")
        nearest_row, nearest_col = divmod(nearest_idx, ds.latitude.shape[1])

        point_slr = ds.slr[..., nearest_row, nearest_col]
        return point_slr
