from dan_weather_suite.models.loader import ModelLoader
import dan_weather_suite.plotting.regions as regions
import dan_weather_suite.utils as utils
from datetime import datetime, time, timedelta
import logging
import numpy as np
from numpy.typing import NDArray
import os
import pickle
from scipy.spatial import KDTree
from typing import Tuple
import time as ttime
import xarray as xr

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


class HrrrLoader(ModelLoader):
    def __init__(self):
        super().__init__()
        self.step_size = 1
        self.forecast_length = 48
        self.forecast_hours = list(
            range(1, self.forecast_length + self.step_size, self.step_size)
        )
        self.grib_file = "grib/hrrr.grib"
        self.netcdf_file = "grib/hrrr.nc"
        self.kd_tree = "hrrr-tree.pkl"

    def get_latest_init(self) -> datetime:
        current_utc = datetime.utcnow()
        current_utc_time = current_utc.time()

        release_00z = time(2, 15)
        release_12z = time(14, 15)

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
        fhour_str = str(fhour).zfill(2)

        extent = regions.PRISM_EXTENT
        left = extent.left
        right = extent.right
        top = extent.top
        bottom = extent.bottom

        base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl"

        params = {
            "dir": f"/hrrr.{day_str}/conus",
            "file": f"hrrr.t{cycle_str}z.wrfsfcf{fhour_str}.grib2",
            "var_APCP": "on",
            #"var_ASNOW": "on",
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

    def process_grib(self) -> xr.Dataset:
        ds = xr.open_dataset(self.grib_file)
        ds["longitude"] = (ds["longitude"] + 180) % 360 - 180
        #ds = ds.rename({"unknown": "asnow"})
        #ds["asnow"].attrs["units"] = "m"
        #ds["asnow].attrs["GRIB_units"] = "m"
        return ds

    def create_kdtree(self) -> KDTree:
        "Create KD tree of ds coordinates"
        ds = self.open_dataset()
        pairs = np.dstack((ds.longitude, ds.latitude)).reshape(-1, 2)
        tree = KDTree(pairs)
        with open("hrrr-tree.pkl", "wb") as f:
            pickle.dump(tree, f)

        return tree

    def get_kdtree(self) -> KDTree:
        "Create KD tree of ds coordinates"
        if not os.path.exists(self.kd_tree):
            tree = self.create_kdtree()
        else:
            with open(self.kd_tree, "rb") as f:
                tree = pickle.load(f)
        return tree


def tst(lon, lat):
    hrrr = HrrrLoader()
    ds = hrrr.open_dataset()

    k_nearest = 16
    if not os.path.exists("hrrr-tree.pkl"):
        tree = self.create_kdtree()
    else:
        with open("hrrr-tree.pkl", "rb") as f:
            tree = pickle.load(f)

    pairs = np.dstack((ds.longitude, ds.latitude)).reshape(-1, 2)
    distances, indicies = tree.query([lon, lat], k_nearest)
    nearest_points = pairs[indicies]

    nearest_coords = [divmod(idx, ds.latitude.shape[1]) for idx in indicies]

    dss = ds.isel(step=-1)

    lats = [dss.sel(y=y, x=x).latitude.values for y, x in nearest_coords]
    lons = [dss.sel(y=y, x=x).longitude.values for y, x in nearest_coords]
    precip = [dss.sel(y=y, x=x).tp.values for y, x in nearest_coords]
    # print(precip)

    from scipy.interpolate import griddata

    data_points = [(lon, lat) for lon, lat in zip(lons, lats)]
    interped = griddata(data_points, precip, [(lon, lat)], method="linear")
    print(interped)
    import matplotlib.pyplot as plt

    plt.tricontourf(lons, lats, precip, levels=10)
    plt.colorbar()

    plt.plot(lon, lat, markersize=2, marker="o", color="k")

    plt.plot(
        nearest_points[0][0],
        nearest_points[0][1],
        markersize=2,
        marker="o",
        color="red",
    )

    plt.show()
