import bz2
import os
from datetime import datetime, time, timedelta
from typing import Tuple

import numpy as np
import xarray as xr
from scipy.interpolate import LinearNDInterpolator

import dan_weather_suite.plotting.regions as regions
import dan_weather_suite.utils as utils
from dan_weather_suite.models.loader import ModelLoader


class IconLoader(ModelLoader):
    def __init__(self):
        super().__init__()
        self.forecast_length = 180
        forecast_hours_6 = list(range(0, 126, 6))
        forecast_hours_12 = list(range(132, 180 + 12, 12))
        self.forecast_hours = forecast_hours_6 + forecast_hours_12
        self.grib_file = "grib/icon.grib"
        self.netcdf_file = "grib/icon.nc"

        self.lons_path = (
            "grib/icon-eps_global_icosahedral_time" "-invariant_2024032400_clon.grib2"
        )
        self.lats_path = (
            "grib/icon-eps_global_icosahedral_time" "-invariant_2024032400_clat.grib2"
        )

    def get_latest_init(self) -> datetime:
        current_utc = datetime.utcnow()
        current_utc_time = current_utc.time()

        release_00z = time(3, 30)
        release_12z = time(15, 30)

        if release_00z <= current_utc_time < release_12z:
            return datetime(current_utc.year, current_utc.month, current_utc.day, 0, 0)
        elif current_utc_time >= release_12z:
            return datetime(current_utc.year, current_utc.month, current_utc.day, 12, 0)
        else:
            previous_day = current_utc - timedelta(days=1)
            return datetime(
                previous_day.year, previous_day.month, previous_day.day, 12, 0
            )

    def download_grib(self, cycle=None):
        init_dt = self.get_latest_init()
        if cycle is not None:
            init_dt = init_dt.replace(hour=cycle)

        urls = [self.url_formatter(init_dt, fhour) for fhour in self.forecast_hours]

        grib_bytes = utils.download_and_combine_gribs(
            urls, compression="bz2", threads=4
        )

        with open(self.grib_file, "wb") as f:
            f.write(grib_bytes)

    def url_formatter(self, init_dt: datetime, fhour) -> Tuple[str, dict]:
        day_str = init_dt.strftime("%Y%m%d%H")
        cycle_str = str(init_dt.hour).zfill(2)
        fhour_str = str(fhour).zfill(3)

        root_url = "https://opendata.dwd.de/weather/nwp/icon-eps/grib"

        return (
            (
                f"{root_url}/{cycle_str}/tot_prec/"
                f"icon-eps_global_icosahedral_single-level_{day_str}_"
                f"{fhour_str}_tot_prec.grib2.bz2"
            ),
            {},
        )

    def download_coordinates(self):

        lons_url = (
            "https://opendata.dwd.de/weather/nwp/icon-eps/"
            "grib/00/clon/icon-eps_global_icosahedral_time"
            "-invariant_2024032400_clon.grib2.bz2"
        )
        lats_url = (
            "https://opendata.dwd.de/weather/nwp/icon-eps"
            "/grib/00/clat/icon-eps_global_icosahedral_time"
            "-invariant_2024032400_clat.grib2.bz2"
        )

        if not os.path.exists(self.lons_path):
            grib_bytes = bz2.decompress(utils.download_bytes(lons_url))
            with open(self.lons_path, "wb") as f:
                f.write(grib_bytes)

        if not os.path.exists(self.lats_path):
            grib_bytes = bz2.decompress(utils.download_bytes(lats_url))
            with open(self.lats_path, "wb") as f:
                f.write(grib_bytes)

    def process_grib(self) -> xr.Dataset:

        self.download_coordinates()
        ds_lats = xr.open_dataset(self.lats_path)
        ds_lons = xr.open_dataset(self.lons_path)
        lats = ds_lats.tlat
        lons = ds_lons.tlon
        ds = xr.open_dataset(self.grib_file, chunks={})
        ds = ds.swap_dims({"step": "valid_time"})

        extent = regions.CONUS_EXTENT

        top = extent.top
        bottom = extent.bottom
        left = extent.left
        right = extent.right

        lon_filter = (lons > left) & (lons < right)
        lat_filter = (lats > bottom) & (lats < top)

        coord_filter = lon_filter & lat_filter

        lons = lons[coord_filter].values
        lats = lats[coord_filter].values

        native_coords = np.column_stack((lons, lats))

        grid_lons = np.arange(left, right, 0.125)
        grid_lats = np.arange(bottom, top, 0.125)

        # our grid to interpolate to
        X, Y = np.meshgrid(grid_lons, grid_lats)

        values = ds.tp.values[:, :, coord_filter]

        # transpose values for required interpolator shape
        interpolator = LinearNDInterpolator(native_coords, values.T)
        Z = interpolator(X, Y)
        # transpose back
        Z = Z.T

        da = xr.DataArray(
            data=Z,
            dims=["number", "valid_time", "longitude", "latitude"],
            coords={
                "number": ds.number.values,
                "valid_time": ds.valid_time.values,
                "latitude": grid_lats,
                "longitude": grid_lons,
                "time": ds.valid_time.values[0],
            },
            attrs=ds.tp.attrs,
        )
        ds_grid = xr.Dataset({"tp": da}, attrs=ds.attrs)
        ds_grid = ds_grid.transpose("number", "valid_time", "latitude", "longitude")
        # ds = ds.sortby(["longitude", "latitude"])
        return ds_grid
