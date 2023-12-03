from dan_weather_suite.models.loader import ModelLoader
import dan_weather_suite.plotting.regions as regions
import dan_weather_suite.utils as utils
from datetime import datetime, time, timedelta
import logging
import numpy as np
import os
import pickle
from scipy.spatial import KDTree
from typing import Tuple
import time as ttime
import xarray as xr

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


class HrefLoader(ModelLoader):
    def __init__(self):
        super().__init__()
        self.step_size = 1
        self.forecast_length = 48
        self.forecast_hours = list(
            range(1, self.forecast_length + self.step_size, self.step_size)
        )
        self.grib_file = "grib/href.grib"
        self.netcdf_file = "grib/href.nc"
        self.products = ["mean", "sprd"]

    def get_latest_init(self) -> datetime:
        current_utc = datetime.utcnow()
        current_utc_time = current_utc.time()

        release_00z = time(3, 20)
        release_12z = time(15, 0)

        if release_00z <= current_utc_time < release_12z:
            return datetime(current_utc.year, current_utc.month, current_utc.day, 0, 0)
        elif current_utc_time >= release_12z:
            return datetime(current_utc.year, current_utc.month, current_utc.day, 12, 0)
        else:
            previous_day = current_utc - timedelta(days=1)
            return datetime(
                previous_day.year, previous_day.month, previous_day.day, 12, 0
            )

    def url_formatter(
        self, init_dt: datetime, fhour: int, product: str
    ) -> Tuple[str, dict]:
        day_str = init_dt.strftime("%Y%m%d")
        cycle_str = str(init_dt.hour).zfill(2)
        fhour_str = str(fhour).zfill(2)

        extent = regions.PRISM_EXTENT
        left = extent.left
        right = extent.right
        top = extent.top
        bottom = extent.bottom

        base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrefconus.pl"

        params = {
            "dir": f"/href.{day_str}/ensprod",
            "file": f"href.t{cycle_str}z.conus.{product}.f{fhour_str}.grib2",
            "var_APCP": "on",
            "subregion": "",
            "toplat": top,
            "leftlon": left,
            "rightlon": right,
            "bottomlat": bottom,
        }

        return base_url, params

    def product_grib_path(self, product: str) -> str:
        return f"grib/href-{product}.grib"

    def download_grib(self, cycle=None):
        "Download each href product into different files"
        init_dt = self.get_latest_init()
        if cycle is not None:
            init_dt = init_dt.replace(hour=cycle)

        # Need to download serially...slowly because of NOAA grib server request limits
        for product in self.products:
            product_grib = self.product_grib_path(product)

            if os.path.exists(product_grib):
                os.remove(product_grib)

            with open(product_grib, "ab") as f:
                for fhour in self.forecast_hours:
                    url, params = self.url_formatter(init_dt, fhour, product)
                    data = utils.download_bytes(url, params)
                    if data:
                        f.write(data)
                        ttime.sleep(0.1)

    def process_grib(self) -> xr.Dataset:
        first_product = self.products[0]
        rest_products = self.products[1:]
        first_product_grib = self.product_grib_path(first_product)

        ds = xr.open_dataset(first_product_grib)
        ds["tp"] = ds.tp.cumsum(dim="step")
        ds = ds.rename({"tp": first_product})

        for product in rest_products:
            product_grib = self.product_grib_path(product)
            ds2 = xr.open_dataset(product_grib)
            ds2["tp"] = ds2.tp.cumsum(dim="step")
            ds2 = ds2.rename({"tp": product})
            ds = ds.merge(ds2)

        ds["longitude"] = (ds["longitude"] + 180) % 360 - 180
        return ds

    def create_kdtree(self) -> KDTree:
        "Create KD tree of ds coordinates"
        ds = self.open_dataset()
        pairs = np.dstack((ds.longitude, ds.latitude)).reshape(-1, 2)
        tree = KDTree(pairs)
        with open("href-tree.pkl", "wb") as f:
            pickle.dump(tree, f)

        return tree
