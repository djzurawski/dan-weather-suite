from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dan_weather_suite.plotting import plot
import dan_weather_suite.plotting.regions as regions
from datetime import datetime, time, timedelta
from dateutil.parser import isoparse
from ecmwf.opendata import Client
import matplotlib.pyplot as plt
import numpy as np
import os
from pydantic import BaseModel, ConfigDict
import tempfile
import time as ttime
import requests
import xarray as xr

IN_PER_M = 39.37
IN_PER_MM = 1 / 25.4
WATER_DENSITY = 1000  # kg/m3


def set_ds_extent(ds, extent=regions.CONUS_EXTENT):
    # left, right, bottom, top = extent
    x_condition = (ds.longitude >= extent.left) & (ds.longitude <= extent.right)
    y_condition = (ds.latitude >= extent.bottom) & (ds.latitude <= extent.top)
    trimmed = ds.where(x_condition & y_condition, drop=True)
    return trimmed


def download_bytes(url: str, from_noaa: bool = False) -> bytes:
    try:
        print("downloading", url)
        resp = requests.get(url)
        if resp.status_code == 200:
            # sleep for NOAA stingy rate limits
            result = resp.content
            if from_noaa:
                ttime.sleep(0.15)
            return result
        else:
            error_str = (
                f"Error downloading {url} status:{resp.status_code}, {resp.text}"
            )
            raise requests.exceptions.RequestException(error_str)

    except requests.exceptions.RequestException as e:
        print(e)
        return None


def download_and_combine_gribs(urls: list[str], threads=4) -> bytes:
    with ThreadPoolExecutor(threads) as executor:
        results = list(executor.map(download_bytes, urls))
        concatenated_bytes = b"".join(
            result for result in results if result is not None
        )
        return concatenated_bytes


def download_and_combine_gribs_from_noaa(urls: list[str]) -> bytes:
    results = [download_bytes(url, True) for url in urls]
    concatenated_bytes = b"".join(result for result in results if result is not None)
    return concatenated_bytes


class EnsembleLoader(ABC):
    def __init__(self):
        self.step_size = 6
        self.forecast_length = 240
        self.forecast_hours = list(
            range(6, self.forecast_length + self.step_size, self.step_size)
        )

    @abstractmethod
    def get_latest_init(self) -> datetime:
        """
        Infers the latest forecat initialization time
        """
        pass

    def is_current(self) -> bool:
        """
        Checks if grib on disk is latest forecast
        """

        ds = xr.open_dataset(self.netcdf_file)
        forecast_init = isoparse(str(ds.time.values))
        latest_init = self.get_latest_init()
        return forecast_init == latest_init

    def download_latest(self, cycle=None):
        if not os.path.exists(self.netcdf_file) or not self.is_current():
            self.fetch_latest_grib()
            ds = self.process_grib()
            ds.to_netcdf(self.netcdf_file)
        print("Up to date")

    @abstractmethod
    def fetch_latest_grib(self, cycle=None):
        """
        Downloads latest forecast
        """
        pass

    def process_grib(self) -> xr.Dataset:
        """
        Loads downloaded grib on disk.
        Combines control and perturbed members into one xr.Dataset
        """
        # chunks = {'step': 40, 'number': 1, 'latitude': 100, 'longitude': 100}
        # chunks = {'step': 'auto', 'number': 'auto', 'latitude': 128, 'longitude': 128}
        ds_c = xr.open_dataset(
            self.grib_file, filter_by_keys={"dataType": "cf"}, chunks={}
        )
        ds_p = xr.open_dataset(
            self.grib_file, filter_by_keys={"dataType": "pf"}, chunks={}
        )

        """
        ds_c = xr.open_mfdataset(
            self.grib_file,
            engine="cfgrib",
            combine="nested",
            concat_dim="step",
            coords="minimal",
            data_vars="minimal",
            compat="override",
            filter_by_keys={"dataType": "cf"},
        )

        # perturbed members
        ds_p = xr.open_mfdataset(
            self.grib_file,
            engine="cfgrib",
            combine="nested",
            concat_dim="step",
            coords="minimal",
            data_vars="minimal",
            compat="override",
            filter_by_keys={"dataType": "pf"},
        )
        """

        # add on the control member to the perturbed
        ds_c_expanded = ds_c.expand_dims("number", axis=1)

        ds = xr.concat([ds_c_expanded, ds_p], "number")
        return ds

    def open_dataset(self):
        return xr.open_dataset(self.netcdf_file)


class PlumeConfiguration(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    loader: EnsembleLoader
    downscale_ratio: xr.Dataset
    plume_color: str
    model: str


class GepsLoader(EnsembleLoader):
    def __init__(self):
        super().__init__()
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

    def fetch_latest_grib(self, cycle=None):
        init_dt = self.get_latest_init()
        urls = [self.url_formatter(init_dt, fhour) for fhour in self.forecast_hours]

        grib_bytes = download_and_combine_gribs(urls)

        with open(self.grib_file, "wb") as f:
            f.write(grib_bytes)

    def url_formatter(self, init_dt: datetime, fhour) -> str:
        day_str = init_dt.strftime("%Y%m%d%H")
        cycle_str = str(init_dt.hour).zfill(2)
        fhour_str = str(fhour).zfill(3)

        root_url = "https://dd.weather.gc.ca/ensemble/geps/grib2/raw"

        return (
            f"{root_url}/{cycle_str}/{fhour_str}/"
            f"CMC_geps-raw_APCP_SFC_0_latlon0p5x0p5_{day_str}_P{fhour_str}_allmbrs"
            ".grib2"
        )

    def process_grib(self):
        ds = super().process_grib()
        ds = ds.rename({"unknown": "tp"})
        ds["tp"].attrs["units"] = "mm"
        ds["tp"].attrs["GRIB_units"] = "mm"
        ds["tp"].attrs["GRIB_shortName"] = "tp"
        ds["tp"].attrs["long_name"] = "Total Precipitation"
        ds["longitude"] = (ds["longitude"] + 180) % 360 - 180
        ds = ds.sortby(["longitude", "latitude"])
        return ds


class EpsLoader(EnsembleLoader):
    def __init__(self):
        self.step_size = 6
        self.forecast_length = 240
        self.client = Client(source="ecmwf")
        self.forecast_hours = list(
            range(0, self.forecast_length + self.step_size, self.step_size)
        )
        self.grib_file = "grib/eps.grib"
        self.netcdf_file = "grib/eps.nc"

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

    def is_current(self) -> bool:
        latest = self.get_latest_init()
        ds = self.open_dataset()
        current = isoparse(str(ds.time.values))

        return latest == current

    def fetch_latest_grib(self):
        latest_init = self.get_latest_init()
        cycle = latest_init.hour
        print("downloading", latest_init)
        self.client.retrieve(
            #time=cycle,
            stream="enfo",
            param="tp",
            step=self.forecast_hours,
            target=self.grib_file,
        )


class GefsLoader(EnsembleLoader):
    def __init__(self):
        self.step_size = 6
        self.forecast_length = 240
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

    def fetch_latest_grib(self):
        init_dt = self.get_latest_init()
        control_member = ["c00"]
        perturbed_members = [f"p{str(i).zfill(2)}" for i in range(1, 31)]
        members = control_member + perturbed_members

        urls = []

        for fhour in self.forecast_hours:
            for member in members:
                urls.append(self.url_formatter(init_dt, fhour, member))
        """
        for member in members:
            for fhour in self.forecast_hours:
                urls.append(self.url_formatter(init_dt, fhour, member))
        """

        grib_bytes = download_and_combine_gribs_from_noaa(urls)
        with open(self.grib_file, "wb") as f:
            f.write(grib_bytes)

    def url_formatter(self, init_dt: datetime, fhour: int, member: str) -> str:
        day_str = init_dt.strftime("%Y%m%d")
        cycle_str = str(init_dt.hour).zfill(2)
        fhour_str = str(fhour).zfill(3)

        return (
            "https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p25s.pl"
            f"?dir=%2Fgefs.{day_str}%2F{cycle_str}%2Fatmos%2Fpgrb2sp25&file="
            f"ge{member}.t{cycle_str}z.pgrb2s.0p25.f{fhour_str}&var_APCP=on"
        )

    def process_grib(self):
        ds = super().process_grib()
        ds["tp"].attrs["long_name"] = "Total Precipitation"
        ds["longitude"] = (ds["longitude"] + 180) % 360 - 180
        ds = ds.sortby(["longitude", "latitude"])
        return ds


def main():
    eps_config = PlumeConfiguration(
        loader=EpsLoader(),
        downscale_ratio=xr.open_dataset("0.4deg-800m.nc"),
        plume_color="green",
        model="EPS",
    )

    geps_config = PlumeConfiguration(
        loader=GepsLoader(),
        downscale_ratio=xr.open_dataset("0.5deg-800m.nc"),
        plume_color="blue",
        model="GEPS",
    )

    gefs_config = PlumeConfiguration(
        loader=GefsLoader(),
        downscale_ratio=xr.open_dataset("0.25deg-800m.nc"),
        plume_color="red",
        model="GEFS",
    )

    ensembles = [eps_config, geps_config, gefs_config]
    #ensembles = [geps_config, gefs_config]
    #ensembles = [gefs_config]

    for ensemble in ensembles:
        #ensemble.loader.download_latest()
        print("pass")

    for location in regions.FRONT_RANGE_LABELS:
    #for location in [regions.Label(text='berthoud', lat=39.8, lon=-105.76)]:
    #for location in [regions.Label(text='berthoud', lat=39.8, lon=-105.77)]:
    #for location in regions.WASATCH_LABELS:
        plt.figure()
        rows = []
        for ensemble in ensembles:
            lat = location.lat
            lon = location.lon

            ds = ensemble.loader.open_dataset()
            precip = ds.tp.interp(latitude=lat, longitude=lon)
            # precip = ds.tp.sel(latitude=lat, longitude=lon, method='nearest')

            if precip.units == "m":
                precip = precip * IN_PER_M
            elif precip.units == "mm":
                precip = precip * IN_PER_MM
            elif precip.units == "kg m**-2":
                precip = (precip / WATER_DENSITY) * IN_PER_M

            precip = precip.transpose("number", "step")

            ratio = ensemble.downscale_ratio.interp(
                latitude=lat, longitude=lon
            ).band_data


            print(ratio.values, location.text, ensemble.model)
            downscaled_precip = ratio * precip

            if ensemble.model == "GEFS":
                downscaled_precip = np.cumsum(downscaled_precip, axis=1)

            for n in precip.number:
                times = downscaled_precip.valid_time.values
                values = downscaled_precip.sel(number=n)

                if ensemble.model != "EPS":
                    t0 = times[0] - np.timedelta64(6, "h")
                    times = np.concatenate([[t0], times])
                    values = np.concatenate([[0], values])

                # rows.append(values)
                plt.plot(
                    times, values, color=ensemble.plume_color, alpha=0.3, linewidth=1
                )

            mean = np.mean(downscaled_precip, axis=0)
            if ensemble.model != "EPS":
                mean = np.insert(mean, 0, 0)

            plt.plot(
                times,
                mean,
                color=ensemble.plume_color,
                linewidth=3,
            )
            plt.title(f"{location.text}, {location.lat}, {location.lon}")
            plt.axvline(times[28], color='gray', linestyle='--')

        """
        plt.figure()
        plt.violinplot(np.array(rows).T.tolist(), showmeans=True, showmedians=True)
        plt.title(location.text)
        plt.figure()
        plt.boxplot(np.array(rows).T.tolist())
        plt.title(location.text)
        """

    plt.show()


def tst():
    ds1 = xr.open_dataset(
        "/home/dan/Downloads/gec00.t00z.pgrb2s.0p25.f135", engine="cfgrib"
    )
    ds2 = xr.open_dataset(
        "/home/dan/Downloads/gep01.t00z.pgrb2s.0p25.f135", engine="cfgrib"
    )
