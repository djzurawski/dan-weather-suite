from dan_weather_suite.models import ModelLoader
import dan_weather_suite.utils as utils
from datetime import datetime, time, timedelta
from typing import Tuple
import xarray as xr


class GepsLoader(ModelLoader):
    def __init__(self):
        super().__init__()
        self.forecast_hours = list(
            range(6, self.forecast_length + self.step_size, self.step_size)
        )
        self.grib_file = "grib/geps.grib"
        self.netcdf_file = "grib/geps.nc"

    def get_latest_init(self) -> datetime:
        current_utc = datetime.utcnow()
        current_utc_time = current_utc.time()

        release_00z = time(4, 45)
        release_12z = time(16, 45)

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

        grib_bytes = utils.download_and_combine_gribs(urls)

        with open(self.grib_file, "wb") as f:
            f.write(grib_bytes)

    def url_formatter(self, init_dt: datetime, fhour) -> Tuple[str, dict]:
        day_str = init_dt.strftime("%Y%m%d%H")
        cycle_str = str(init_dt.hour).zfill(2)
        fhour_str = str(fhour).zfill(3)

        root_url = "https://dd.weather.gc.ca/ensemble/geps/grib2/raw"

        return (
            (
                f"{root_url}/{cycle_str}/{fhour_str}/"
                f"CMC_geps-raw_APCP_SFC_0_latlon0p5x0p5_{day_str}_P{fhour_str}_allmbrs"
                ".grib2"
            ),
            root_url,
        )

    def process_grib(self) -> xr.Dataset:
        ds = super().process_grib()
        ds = ds.rename({"unknown": "tp"})
        ds["tp"].attrs["units"] = "mm"
        ds["tp"].attrs["GRIB_units"] = "mm"
        ds["tp"].attrs["GRIB_shortName"] = "tp"
        ds["tp"].attrs["long_name"] = "Total Precipitation"
        ds["longitude"] = (ds["longitude"] + 180) % 360 - 180
        ds = ds.sortby(["longitude", "latitude"])
        return ds
