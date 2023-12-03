from dan_weather_suite.models import ModelLoader
from datetime import datetime, time
from dateutil.parser import isoparse
from ecmwf.opendata import Client
import logging

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


class EpsLoader(ModelLoader):
    def __init__(self):
        super().__init__()
        self.forecast_hours = list(
            range(0, self.forecast_length + self.step_size, self.step_size)
        )
        self.client = Client(source="ecmwf")
        self.grib_file = "grib/eps.grib"
        self.netcdf_file = "grib/eps.nc"
        self.z12_ready_time_utc = time(20, 30)
        self.z00_ready_time_utc = time(8, 30)

    def get_latest_init(self) -> datetime:
        latest = self.client.latest(
            stream="enfo",
        )
        # we're only use cycle 00 and 12
        if latest.hour == 18:
            latest = latest.replace(hour=12)
        if latest.hour == 6:
            latest = latest.replace(hour=0)

        return latest

    def is_current(self, cycle=None) -> bool:
        latest = self.get_latest_init()
        ds = self.open_dataset()
        current = isoparse(str(ds.time.values))

        if cycle is not None:
            latest = latest.replace(hour=cycle)

        return latest == current

    def download_grib(self, cycle=None):
        latest_init = self.get_latest_init()
        if cycle is None:
            cycle = latest_init.hour
        logging.info(f"EPS downloading {cycle}")
        self.client.retrieve(
            time=cycle,
            stream="enfo",
            param="tp",
            step=self.forecast_hours,
            target=self.grib_file,
        )
