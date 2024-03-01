from concurrent.futures import ProcessPoolExecutor
from dan_weather_suite.models.loader import ModelLoader
from dan_weather_suite.models.gefs import GefsLoader
from dan_weather_suite.models.geps import GepsLoader
from dan_weather_suite.models.eps import EpsLoader
from dan_weather_suite.models.nbm import NbmLoader
from dan_weather_suite.plotting import plot
import dan_weather_suite.utils as utils
import dan_weather_suite.plotting.regions as regions
import dask
from datetime import datetime
import io
import logging
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import xarray as xr
import traceback
from typing import Literal, Iterable

dask.config.set({"array.slicing.split_large_chunks": True})

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
logger = logging.getLogger("ensemble")

EnsembleName = Literal["GEFS", "CMCE", "ECMWF"]


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

        downscale_ds_number = utils.round_to_nearest(resolution_deg)
        downscale_ds_file = f"{downscale_ds_number}deg-800m.nc"
        return xr.open_dataset(downscale_ds_file)

    def _forecast_starts_at_init(self) -> bool:
        "Some models the first data is at init_time + step_size"
        return self.ds.valid_time.values[0] == self.ds.time.values

    def point_plumes(
        self, lon: float, lat: float, downscale=True, nearest=False, accum_snow=False
    ) -> Tuple[NDArray[np.datetime64], NDArray[float]]:
        ds = self.ds
        prepend_t0 = not self._forecast_starts_at_init()

        units = ds.tp.units
        conversion = utils.swe_to_in(units)

        if nearest:
            precip = conversion * ds.tp.sel(
                latitude=lat, longitude=lon, method="nearest"
            )
        else:
            precip = conversion * ds.tp.interp(latitude=lat, longitude=lon)

        times = precip.valid_time.values
        # add t0 to ensembles which forecast starts at 6h
        if prepend_t0:
            t0 = times[0] - np.timedelta64(6, "h")
            times = np.concatenate([[t0], times])

        if nearest and downscale:
            model_resolution_deg = np.abs(np.diff(sorted(ds.latitude)))[0]
            center_lat = precip.latitude
            center_lon = precip.longitude

            top_lat = center_lat + (model_resolution_deg / 2)
            bottom_lat = center_lat - (model_resolution_deg / 2)
            left_lon = center_lon - (model_resolution_deg / 2)
            right_lon = center_lon + (model_resolution_deg / 2)
            extent = regions.Extent(
                top=top_lat, bottom=bottom_lat, left=left_lon, right=right_lon
            )
            grid_prism_ds = utils.set_ds_extent(self.downscale_ds, extent)
            grid_mean = grid_prism_ds.prism.mean()
            point = self.downscale_ds.prism.interp(latitude=lat, longitude=lon)
            ratio = point / grid_mean
            precip = ratio * precip

        elif downscale:
            ratio = self.downscale_ds.interp(latitude=lat, longitude=lon).ratio
            logger.info(f"Ratio {ratio.values} at {lat},{lon}")
            precip = ratio * precip

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

        conversion = utils.swe_to_in(self.ds.tp.units)
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
                precip_points = self.downscale_ds.ratio * precip_points

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


def plot_compare(ens: Ensemble, fhour=84):
    lon, lat, swe = ens.swe_at_fhour(fhour, downscale=True, ratio=True)
    plot.plot_swe(lon, lat, swe, pcolormesh=True)
    plt.title("Downscaled")

    lon, lat, swe = ens.swe_at_fhour(fhour, downscale=True, ratio=False)
    plot.plot_swe(lon, lat, swe, pcolormesh=True)
    plt.title("Interpolated")

    lon, lat, swe = ens.swe_at_fhour(fhour, downscale=False, ratio=False)
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


def get_point_plumes(ensemble, slr_ds, lon, lat, downscale=True, nearest=False):
    times, precip_plumes = ensemble.point_plumes(
        lon, lat, downscale=downscale, nearest=nearest
    )

    # no t0 in nbm
    slr_steps = [t - times[0] for t in times[1:]]
    slr = slr_ds.interp(step=slr_steps).values

    snow_plumes = np.zeros(precip_plumes.shape)
    precip_rate = np.diff(precip_plumes, axis=1)
    snow_rate = precip_rate * slr
    snow_plumes[:, 1:] = np.cumsum(snow_rate, axis=1)

    precip_mean = np.mean(precip_plumes, axis=0)
    snow_mean = np.mean(snow_plumes, axis=0)

    return times, precip_plumes, snow_plumes, precip_mean, snow_mean


def add_ensemble_plumes_to_plot(
    axs,
    times,
    precip_plumes,
    snow_plumes,
    precip_mean,
    snow_mean,
    name: str,
    color,
    downscale=True,
    nearest=False,
):

    for plume, snow_plume in zip(precip_plumes, snow_plumes):
        axs[0, 0].plot(times, plume, color=color, alpha=0.3, linewidth=1)

        axs[1, 0].plot(times, snow_plume, color=color, alpha=0.3, linewidth=1)

    axs[0, 0].plot(times, precip_mean, color=color, linewidth=3, zorder=200, label=name)
    axs[1, 0].plot(
        times,
        snow_mean,
        color=color,
        linewidth=3,
        zorder=200,
        label=name,
    )

    return axs


def plume_plot_snow(
    lon,
    lat,
    title="",
    models: Iterable[EnsembleName] = ["GEFS", "CMCE", "ECMWF"],
    downscale=True,
    nearest=False,
    return_bytes: bool = False,
):
    LOADERS = {
        "GEFS": (GefsLoader(), "GEFS", "red"),
        "CMCE": (GepsLoader(), "CMCE", "blue"),
        "ECMWF": (EpsLoader(), "ECMWF ENS", "green"),
    }

    nbm = NbmLoader()
    slr_ds = nbm.forecast_slr(lon, lat)

    fig, axs = plt.subplots(2, 2, figsize=(16, 10), sharey="row")
    fig.suptitle(f"{title} lat: {lat} lon: {lon}")
    plt.tight_layout(pad=3)

    all_precip = []
    all_snow = []
    for model in models:
        try:
            ensemble = Ensemble(*LOADERS[model])
            times, precip_plumes, snow_plumes, precip_mean, snow_mean = (
                get_point_plumes(ensemble, slr_ds, lon, lat, downscale, nearest)
            )

            # no t0 in nbm
            slr_steps = [t - times[0] for t in times[1:]]
            slr = slr_ds.interp(step=slr_steps).values

            [all_precip.append(plume) for plume in precip_plumes]
            [all_snow.append(plume) for plume in snow_plumes]

            axs = add_ensemble_plumes_to_plot(
                axs,
                times,
                precip_plumes,
                snow_plumes,
                precip_mean,
                snow_mean,
                ensemble.name,
                ensemble.plume_color,
            )

        except Exception as e:
            logger.error(f"Error plotting ensemble {model}: {e}")
            traceback.print_exc()

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
    times_labels = [xtick_formatter(utils.parse_np_datetime64(t)) for t in times_ticks]
    axs[0, 0].set_xticks(times_ticks, labels=times_labels)
    axs[1, 0].set_xticks(times_ticks, labels=times_labels)

    axs[0, 1].set_xticks(np.arange(1, len(times) + 1, 2), labels=times_labels)
    axs[1, 1].set_xticks(np.arange(1, len(times) + 1, 2), labels=times_labels)

    # vertical line at day 7
    day_7 = int(168 / 6)
    axs[0, 0].axvline(times[day_7], color="gray", linestyle="--")
    axs[1, 0].axvline(times[day_7], color="gray", linestyle="--")
    axs[0, 1].axvline(day_7 + 1, color="gray", linestyle="--")
    axs[1, 1].axvline(day_7 + 1, color="gray", linestyle="--")

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
