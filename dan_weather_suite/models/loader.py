from abc import ABC, abstractmethod
from datetime import datetime, time
import dan_weather_suite.plotting.regions as regions
import dan_weather_suite.utils as utils
from dateutil.parser import isoparse
import logging
import os
import time as ttime
import xarray as xr

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


class ModelLoader(ABC):
    def __init__(self):
        self.step_size = 6
        self.forecast_length = 240

    @abstractmethod
    def get_latest_init(self) -> datetime:
        """
        Infers the latest forecat initialization time
        """
        pass

    def is_current(self, cycle=None) -> bool:
        """
        Checks if grib on disk is latest forecast
        """

        ds = xr.open_dataset(self.netcdf_file, chunks={})
        forecast_init = isoparse(str(ds.time.values))
        latest_init = self.get_latest_init()
        if cycle is not None:
            latest_init = latest_init.replace(hour=cycle)
        return forecast_init == latest_init

    def download_forecast(self, cycle=None, force=False):
        logging.info("Downloading grib")

        if force:
            if os.path.exists(self.grib_file):
                # delete netcdf later to preserve website uptime
                os.remove(self.grib_file)

            if os.path.exists(self.netcdf_file):
                # delete netcdf later to preserve website uptime
                os.remove(self.netcdf_file)

        if not os.path.exists(self.netcdf_file) or not self.is_current(cycle):
            logging.info(f"Downloading grib {cycle}")
            self.download_grib(cycle)

        logging.info("Processing grib")
        ds = self.process_grib()
        logging.info("Setting CONUS extent")
        extent = regions.PRISM_EXTENT
        ds = utils.set_ds_extent(ds, extent)
        retries = 0
        while retries <= 3:
            try:
                logging.info("Saving to NETCDF")
                if force and os.path.exists(self.netcdf_file):
                    os.remove(self.netcdf_file)
                ds.to_netcdf(self.netcdf_file)
                logging.info("Up to date")
                return True
            except Exception as e:
                logging.error(f"Error saving NETCDF {self.netcdf_file}: {e}")
                ttime.sleep(3)
                retries += 1

        return False

    @abstractmethod
    def download_grib(self):
        """
        Downloads latest forecast
        """
        pass

    def process_grib(self) -> xr.Dataset:
        """
        Loads downloaded grib on disk.
        Combines control and perturbed members into one xr.Dataset
        """

        ds_c = xr.open_dataset(
            self.grib_file, filter_by_keys={"dataType": "cf"}, chunks={}
        )
        ds_p = xr.open_dataset(
            self.grib_file, filter_by_keys={"dataType": "pf"}, chunks={"number": 1}
        )

        # add on the control member to the perturbed
        ds_c_expanded = ds_c.expand_dims("number", axis=1)

        ds = xr.concat([ds_c_expanded, ds_p], "number")
        ds = ds.swap_dims({"step": "valid_time"})

        return ds

    def open_dataset(self) -> xr.Dataset:
        return xr.open_dataset(self.netcdf_file)
