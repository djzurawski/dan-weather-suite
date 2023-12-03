from dan_weather_suite.models import ModelLoader
import dan_weather_suite.plotting.regions as regions
import dan_weather_suite.utils as utils
from datetime import datetime, time, timedelta
from typing import Tuple
import time as ttime
import xarray as xr


class GefsLoader(ModelLoader):
    def __init__(self):
        super().__init__()
        self.forecast_hours = list(
            range(6, self.forecast_length + self.step_size, self.step_size)
        )
        self.grib_file = "grib/gefs.grib"
        self.netcdf_file = "grib/gefs.nc"

    def get_latest_init(self) -> datetime:
        current_utc = datetime.utcnow()
        current_utc_time = current_utc.time()

        release_00z = time(5, 40)
        release_12z = time(17, 30)

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
        control_member = ["c00"]
        perturbed_members = [f"p{str(i).zfill(2)}" for i in range(1, 31)]
        members = control_member + perturbed_members

        # urls = []
        # Need to download serially...slowly because of NOAA grib server request limits
        grib_bytes = []
        for fhour in self.forecast_hours:
            for member in members:
                url, params = self.url_formatter(init_dt, fhour, member)
                # urls.append(self.url_formatter(init_dt, fhour, member))
                grib_bytes.append(utils.download_bytes(url, params))
                ttime.sleep(0.1)

        concatenated_bytes = b"".join(
            result for result in grib_bytes if result is not None
        )
        with open(self.grib_file, "wb") as f:
            f.write(concatenated_bytes)

    def url_formatter(
        self, init_dt: datetime, fhour: int, member: str
    ) -> Tuple[str, dict]:
        day_str = init_dt.strftime("%Y%m%d")
        cycle_str = str(init_dt.hour).zfill(2)
        fhour_str = str(fhour).zfill(3)

        extent = regions.PRISM_EXTENT
        left = extent.left
        right = extent.right
        top = extent.top
        bottom = extent.bottom

        base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p25s.pl"

        params = {
            "dir": f"/gefs.{day_str}/{cycle_str}/atmos/pgrb2sp25",
            "file": f"ge{member}.t{cycle_str}z.pgrb2s.0p25.f{fhour_str}",
            "var_APCP": "on",
            "subregion": "",
            "toplat": top,
            "leftlon": left,
            "rightlon": right,
            "bottomlat": bottom,
        }

        return base_url, params

    def process_grib(self) -> xr.Dataset:
        ds = super().process_grib()
        ds["tp"] = ds.tp.cumsum(dim="step")
        ds["tp"].attrs["long_name"] = "Total Precipitation"
        ds["tp"].attrs["units"] = "kg m**-2"
        ds["longitude"] = (ds["longitude"] + 180) % 360 - 180
        ds = ds.sortby(["longitude", "latitude"])
        return ds
