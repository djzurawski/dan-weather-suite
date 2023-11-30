from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections.abc import Iterable
from dan_weather_suite.plotting import plot
import dan_weather_suite.plotting.regions as regions
import dask
from datetime import datetime, time, timedelta, timezone
from dateutil.parser import isoparse
from ecmwf.opendata import Client
import io
import logging
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import os
import pickle
import requests
from scipy.spatial import KDTree
import time as ttime
from typing import Tuple
import xarray as xr


dask.config.set({"array.slicing.split_large_chunks": True})

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
logger = logging.getLogger("ensemble")


def parse_np_datetime64(t: np.datetime64) -> datetime:
    return isoparse(str(t)).replace(tzinfo=timezone.utc)


def round_to_nearest(x: float, options: Iterable[float] = [0.25, 0.4, 0.5]) -> float:
    "Rounds x to the nearest value in 'options'"
    assert options, "options cannot be empty"

    closest = min(options, key=lambda opt: abs(opt - x))
    return closest


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371  # Earth radius in kilometers

    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat / 2) * np.sin(dLat / 2) + np.cos(np.radians(lat1)) * np.cos(
        np.radians(lat2)
    ) * np.sin(dLon / 2) * np.sin(dLon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance


def swe_to_in(units: str) -> float:
    IN_PER_M = 39.37
    IN_PER_MM = 1 / 25.4
    WATER_DENSITY = 1000  # kg/m3

    if units == "m":
        conversion = IN_PER_M
    elif units == "mm":
        conversion = IN_PER_MM
    elif units == "kg m**-2":
        conversion = (1 / WATER_DENSITY) * IN_PER_M
    else:
        raise ValueError(f"Unimplemented Unit conversion: {units} to in")

    return conversion


def set_ds_extent(ds: xr.Dataset, extent: regions.Extent) -> xr.Dataset:
    left = extent.left
    right = extent.right
    top = extent.top
    bottom = extent.bottom
    x_condition = (ds.longitude >= left) & (ds.longitude <= right)
    y_condition = (ds.latitude >= bottom) & (ds.latitude <= top)
    trimmed = ds.where(x_condition & y_condition, drop=True)
    return trimmed


def download_bytes(url: str, params: dict = {}) -> bytes:
    try:
        logger.info(f"downloading {url} {params}")
        resp = requests.get(url, params=params, timeout=600)
        if resp.status_code == 200:
            result = resp.content
            return result
        else:
            error_str = logger.error(
                f"Error downloading {url} status:{resp.status_code}, {resp.text}"
            )
            raise requests.exceptions.RequestException(error_str)

    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading {url} status:{resp.status_code}, {e}")
        return None


def download_and_combine_gribs(urls: list[Tuple[str, dict]], threads=1) -> bytes:
    with ThreadPoolExecutor(threads) as executor:
        results = list(executor.map(lambda x: download_bytes(*x), urls))
        concatenated_bytes = b"".join(
            result for result in results if result is not None
        )
        return concatenated_bytes


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

        ds = xr.open_dataset(self.netcdf_file)
        forecast_init = isoparse(str(ds.time.values))
        latest_init = self.get_latest_init()
        if cycle is not None:
            latest_init = latest_init.replace(hour=cycle)
        return forecast_init == latest_init

    def download_forecast(self, cycle=None, force=False):
        logger.info("Downloading grib")
        if force:
            if os.path.exists(self.netcdf_file):
                os.remove(self.netcdf_file)
            if os.path.exists(self.grib_file):
                os.remove(self.grib_file)

        if not os.path.exists(self.netcdf_file) or not self.is_current(cycle):
            logger.info(f"Downloading grib {cycle}")
            self.download_grib(cycle)
            logger.info("Processed grib")
            ds = self.process_grib()
            logger.info("Setting CONUS extent")
            extent = regions.PRISM_EXTENT
            ds = set_ds_extent(ds, extent)
            retries = 0
            while retries <= 3:
                try:
                    logger.info("Saving to NETCDF")
                    ds.to_netcdf(self.netcdf_file, mode="a")
                    logger.info("Up to date")
                    return True
                except Exception as e:
                    logger.error(f"Error saving NETCDF {self.netcdf_file}: {e}")
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
            self.grib_file, filter_by_keys={"dataType": "pf"}, chunks={}
        )

        # add on the control member to the perturbed
        ds_c_expanded = ds_c.expand_dims("number", axis=1)

        ds = xr.concat([ds_c_expanded, ds_p], "number")

        return ds

    def open_dataset(self) -> xr.Dataset:
        return xr.open_dataset(self.netcdf_file)


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

        grib_bytes = download_and_combine_gribs(urls)

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
        logger.info(f"EPS downloading {cycle}")
        self.client.retrieve(
            time=cycle,
            stream="enfo",
            param="tp",
            step=self.forecast_hours,
            target=self.grib_file,
        )


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
                grib_bytes.append(download_bytes(url, params))
                ttime.sleep(0.2)

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


class NbmLoader(ModelLoader):
    def __init__(self):
        super().__init__()
        self.forecast_hours = list(
            range(6, self.forecast_length + self.step_size, self.step_size)
        )
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
        grib_bytes = []
        for fhour in self.forecast_hours:
            url, params = self.url_formatter(
                init_dt,
                fhour,
            )
            grib_bytes.append(download_bytes(url, params))
            ttime.sleep(0.2)

        concatenated_bytes = b"".join(
            result for result in grib_bytes if result is not None
        )
        with open(self.grib_file, "wb") as f:
            f.write(concatenated_bytes)

    def process_grib(self) -> xr.Dataset:
        ds = xr.open_dataset(self.grib_file)
        ds = ds.rename({"unknown": "slr"})
        ds["longitude"] = (ds["longitude"] + 180) % 360 - 180
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
            (haversine(lat, lon, nbm_lat, nbm_lon), idx)
            for (nbm_lon, nbm_lat), idx in zip(nearest_points, indicies)
        ]

        nearest_point_km, nearest_idx = min(nearest_points_idx, key=lambda x: x[0])
        logger.info(f"Nearest NBM: {pairs[nearest_idx]} {round(nearest_point_km,2)}km")

        nearest_row, nearest_col = divmod(nearest_idx, ds.latitude.shape[1])
        return ds.slr[..., nearest_row, nearest_col]


class Ensemble:
    def __init__(self, loader: ModelLoader, name="Ens", plume_color="red"):
        self.loader = loader
        self.name = name
        self.plume_color = plume_color
        self.ds = loader.open_dataset()
        self.downscale_ds = self._select_downscale_dataset()

    def _select_downscale_dataset(self) -> xr.Dataset:
        forecast_ds = self.loader.open_dataset()
        resolution_deg = np.abs(np.diff(forecast_ds.latitude))[0]

        downscale_ds_number = round_to_nearest(resolution_deg)
        downscale_ds_file = f"{downscale_ds_number}deg-800m.nc"
        return xr.open_dataset(downscale_ds_file)

    def _forecast_starts_at_init(self) -> bool:
        "Some models the first data is at init_time + step_size"
        return self.ds.valid_time.values[0] == self.ds.time.values

    def point_plumes(
        self, lon: float, lat: float, downscale=True, accum_snow=False
    ) -> Tuple[NDArray[np.datetime64], NDArray[float]]:
        ds = self.ds
        prepend_t0 = not self._forecast_starts_at_init()

        units = ds.tp.units
        conversion = swe_to_in(units)
        precip = ds.tp.interp(latitude=lat, longitude=lon)

        times = precip.valid_time.values
        # add t0 to ensembles which forecast starts at 6h
        if prepend_t0:
            t0 = times[0] - np.timedelta64(6, "h")
            times = np.concatenate([[t0], times])

        if downscale:
            ratio = self.downscale_ds.interp(latitude=lat, longitude=lon).band_data
            logger.info(f"Ratio {ratio.values} at {lat},{lon}")
            precip = ratio * precip * conversion

        plumes = []
        for n in precip.number:
            values = precip.sel(number=n).to_numpy()
            if prepend_t0:
                values = np.concatenate([[0], values])

            plumes.append(values)

        plumes = np.array(plumes)

        if accum_snow:
            nbm = NbmLoader()
            slr = nbm.forecast_slr(lon, lat).values
            snow_rate = np.diff(plumes, axis=1) * slr
            cumsnow = np.cumsum(snow_rate, axis=1)
            plumes = cumsnow

        return times, np.array(plumes)

    def swe_at_fhour(
        self,
        fhour: int,
        downscale=True,
        member: int | None = None,
        percentile: float | None = None,
        ratio: bool = True,
    ) -> Tuple[NDArray[float], NDArray[float], xr.DataArray]:
        step_size = self.loader.step_size
        assert fhour % step_size == 0, "forecast hour must be divisible by 6"

        conversion = swe_to_in(self.ds.tp.units)
        step = int(fhour / step_size)
        if not self._forecast_starts_at_init:
            step -= 1

        # We need to filter down the dataset before computing
        # To not blow up memory
        ds_step = self.ds.isel(step=step)
        if member:
            precip = conversion * ds_step.tp.isel(number=member)
        elif percentile:
            precip = conversion * ds_step.tp.quantile(percentile, dim="number")
        else:
            precip = conversion * ds_step.tp.mean(dim="number")

        if not downscale:
            return self.ds.longitude, self.ds.latitude, precip
        else:
            precip_points = precip.interp(
                latitude=self.downscale_ds.latitude,
                longitude=self.downscale_ds.longitude,
                method="linear",
            )

            if ratio:
                precip_points = self.downscale_ds.band_data * precip_points

            return (
                precip_points.longitude.values,
                precip_points.latitude.values,
                precip_points,
            )


def download_loader_forecast(loader_class: type[ModelLoader], cycle=None, force=False):
    return loader_class().download_forecast(cycle, force)


def download_all_forecasts(cycle=None, force=False):
    with ProcessPoolExecutor() as executor:
        gefs = executor.submit(download_loader_forecast, GefsLoader, cycle, force=force)
        geps = executor.submit(download_loader_forecast, GepsLoader, cycle, force=force)
        eps = executor.submit(download_loader_forecast, EpsLoader, cycle, force=force)

        futures = [gefs, geps, eps]
        ensemble_res = [f.result() for f in futures]

        nbm = NbmLoader()
        nbm_res = nbm.download_forecast(cycle=cycle, force=force)
        return ensemble_res + [nbm_res]


def plot_compare(ens: Ensemble):
    lon, lat, swe = ens.swe_at_fhour(84, downscale=True, ratio=True)
    plot.plot_swe(lon, lat, swe, pcolormesh=True)
    plt.title("Downscaled")

    lon, lat, swe = ens.swe_at_fhour(84, downscale=True, ratio=False)
    plot.plot_swe(lon, lat, swe, pcolormesh=True)
    plt.title("Interpolated")

    lon, lat, swe = ens.swe_at_fhour(84, downscale=False, ratio=False)
    plot.plot_swe(lon, lat, swe, pcolormesh=True)
    plt.title("Native")
    # plt.show()


def xtick_formatter(dt: datetime):
    if dt.hour == 12:
        return "12z"
    if dt.hour == 0:
        return "00z\n" + dt.strftime("%b-%d")
    else:
        return ""


def plume_plot(lon, lat, title="", models=[], return_bytes: bool = False):
    LOADERS = {
        "GEFS": (GefsLoader(), "GEFS", "red"),
        "CMCE": (GepsLoader(), "CMCE", "blue"),
        "ECMWF": (EpsLoader(), "ECMWF ENS", "green"),
    }

    if not models:
        ensembles = [Ensemble(*loader) for loader in LOADERS.values()]
    else:
        ensembles = [Ensemble(*LOADERS[model]) for model in models]

    fig, axs = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    plt.tight_layout(pad=2)

    all_plumes = []
    for ensemble in ensembles:
        times, plumes = ensemble.point_plumes(lon, lat)
        for plume in plumes:
            axs[0].plot(
                times, plume, color=ensemble.plume_color, alpha=0.3, linewidth=1
            )
            all_plumes.append(plume)

        mean = np.mean(plumes, axis=0)
        axs[0].plot(
            times,
            mean,
            color=ensemble.plume_color,
            linewidth=3,
            zorder=200,
            label=ensemble.name,
        )

    axs[0].legend()

    boxplot_data = np.array(all_plumes)
    axs[1].boxplot(boxplot_data, showfliers=False, whis=(10, 90))
    """
    axs[1].violinplot(
        boxplot_data,
        showmeans=True,
        showmedians=True,
        showextrema=False,
        widths=0.8,
    )
    """

    times_ticks = times[::2]
    times_labels = [xtick_formatter(parse_np_datetime64(t)) for t in times_ticks]
    axs[0].set_xticks(times_ticks, labels=times_labels)
    axs[1].set_xticks(np.arange(1, len(times) + 1, 2), labels=times_labels)

    fig.suptitle(f"{title} precipitation (in) lat: {lat} lon: {lon}")

    # vertical line at day 7
    day_7 = int(168 / 6)
    axs[0].axvline(times[day_7], color="gray", linestyle="--")
    axs[1].axvline(day_7, color="gray", linestyle="--")

    # turn second y axis labels on
    axs[1].yaxis.set_tick_params(labelleft=True)

    axs[0].grid(axis="both", linestyle="--")
    axs[1].grid(axis="both", linestyle="--")

    if return_bytes:
        with io.BytesIO() as bio:
            plt.savefig(bio, format="jpg", bbox_inches="tight")
            return bio.getvalue()

    plt.show()


def plume_plot_snow(lon, lat, title="", models=[], return_bytes: bool = False):
    LOADERS = {
        "GEFS": (GefsLoader(), "GEFS", "red"),
        "CMCE": (GepsLoader(), "CMCE", "blue"),
        "ECMWF": (EpsLoader(), "ECMWF ENS", "green"),
    }

    if not models:
        ensembles = [Ensemble(*loader) for loader in LOADERS.values()]
    else:
        ensembles = [Ensemble(*LOADERS[model]) for model in models]

    nbm = NbmLoader()
    slr = nbm.forecast_slr(lon, lat)

    fig, axs = plt.subplots(2, 2, figsize=(16, 10), sharey="row")
    fig.suptitle(f"{title} lat: {lat} lon: {lon}")
    plt.tight_layout(
        pad=3,
    )

    all_precip = []
    all_snow = []
    for ensemble in ensembles:
        times, plumes = ensemble.point_plumes(lon, lat)

        snow_plumes = np.zeros(plumes.shape)
        precip_rate = np.diff(plumes, axis=1)
        snow_rate = precip_rate * slr.values
        snow_plumes[:, 1:] = np.cumsum(snow_rate, axis=1)

        for plume, snow_plume in zip(plumes, snow_plumes):
            axs[0, 0].plot(
                times, plume, color=ensemble.plume_color, alpha=0.3, linewidth=1
            )

            axs[1, 0].plot(
                times, snow_plume, color=ensemble.plume_color, alpha=0.3, linewidth=1
            )
            all_precip.append(plume)
            all_snow.append(snow_plume)

        precip_mean = np.mean(plumes, axis=0)
        snow_mean = np.mean(snow_plumes, axis=0)
        axs[0, 0].plot(
            times,
            precip_mean,
            color=ensemble.plume_color,
            linewidth=3,
            zorder=200,
            label=ensemble.name,
        )
        axs[1, 0].plot(
            times,
            snow_mean,
            color=ensemble.plume_color,
            linewidth=3,
            zorder=200,
            label=ensemble.name,
        )

    axs[0, 0].legend()
    axs[1, 0].legend()

    # Boxplots
    precip_boxplot_data = np.array(all_precip)
    snow_boxplot_data = np.array(all_snow)
    axs[0, 1].boxplot(precip_boxplot_data, showfliers=False, whis=(10, 90))
    axs[1, 1].boxplot(snow_boxplot_data, showfliers=False, whis=(10, 90))

    # SLR Line
    ax_slr = axs[1, 1].twinx()
    slr_x = np.arange(2, len(slr) + 2)
    ax_slr.plot(slr_x, slr, color="gray", label="Snow Liquid Ratio")
    ax_slr.set_ylim((0, 30))
    ax_slr.legend()
    ax_slr.set_ylabel("Snow:Liquid Ratio")

    # Set xaxis labels
    times_ticks = times[::2]
    times_labels = [xtick_formatter(parse_np_datetime64(t)) for t in times_ticks]
    axs[0, 0].set_xticks(times_ticks, labels=times_labels)
    axs[1, 0].set_xticks(times_ticks, labels=times_labels)

    axs[0, 1].set_xticks(np.arange(1, len(times) + 1, 2), labels=times_labels)
    axs[1, 1].set_xticks(np.arange(1, len(times) + 1, 2), labels=times_labels)

    # vertical line at day 7
    day_7 = int(168 / 6)
    axs[0, 0].axvline(times[day_7], color="gray", linestyle="--")
    axs[1, 0].axvline(times[day_7], color="gray", linestyle="--")
    axs[0, 1].axvline(day_7, color="gray", linestyle="--")
    axs[1, 1].axvline(day_7, color="gray", linestyle="--")

    # turn second y axis labels on
    axs[0, 1].yaxis.set_tick_params(labelleft=True)
    axs[1, 1].yaxis.set_tick_params(labelleft=True)

    # Grid dotted lines
    axs[0, 0].grid(axis="both", linestyle="--")
    axs[1, 0].grid(axis="both", linestyle="--")
    axs[0, 1].grid(axis="both", linestyle="--")
    axs[1, 1].grid(axis="both", linestyle="--")

    # Subplot titles
    axs[0, 0].title.set_text("Accumulated Precipitation")
    axs[0, 1].title.set_text("Accumulated Precipitation")
    axs[1, 0].title.set_text("Accumulated Snow")
    axs[1, 1].title.set_text("Accumulated Snow")

    # Subplot ylabels
    axs[0, 0].set_ylabel("Precip (in)")
    axs[0, 1].set_ylabel("Precip (in)")
    axs[1, 0].set_ylabel("Snow (in)")
    axs[1, 1].set_ylabel("Snow (in)")

    if return_bytes:
        with io.BytesIO() as bio:
            plt.savefig(bio, format="jpg", bbox_inches="tight")
            plt.close(fig)
            return bio.getvalue()

    plt.show()
