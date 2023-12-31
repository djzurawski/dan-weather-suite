from dan_weather_suite.models.hiresw_arw import (
    HireswArwLoader,
    HireswArw2Loader,
    HireswFv3Loader,
)
from dan_weather_suite.models.hrrr import HrrrLoader
from dan_weather_suite.models.nbm import NbmLoader
from dan_weather_suite.models.rrfs import RrfsLoader
from datetime import datetime
import dan_weather_suite.utils as utils
import io
import numpy as np
import matplotlib.pyplot as plt


def xtick_formatter(dt: datetime):
    if dt.hour == 6:
        return "06z"
    if dt.hour == 12:
        return "12z"
    if dt.hour == 0:
        return "00z\n" + dt.strftime("%b-%d")
    else:
        return ""


def plume_plot(lon: float, lat: float, title="", return_bytes=False):
    LOADERS = {
        "ARW": HireswArwLoader(),
        "ARW2": HireswArw2Loader(),
        "FV3": HireswFv3Loader(),
        "HRRR": HrrrLoader(),
    }

    ENSEMBLE_LOADERS = {"RRFS": RrfsLoader()}

    nbm = NbmLoader(short_term=True)
    slr_ds = nbm.forecast_slr(lon, lat)

    fig, axs = plt.subplots(2, 2, figsize=(16, 10), sharey="row")
    fig.suptitle(f"{title} lat: {lat} lon: {lon}")
    plt.tight_layout(
        pad=3,
    )

    all_precip = []
    all_snow = []
    for model_name, loader in LOADERS.items():
        ds = loader.open_dataset()
        kdtree = loader.get_kdtree()
        point_forecast = utils.nearest_neighbor_forecast(ds, kdtree, lon, lat)

        precip_raw = point_forecast.tp
        conversion = utils.swe_to_in(precip_raw.units)
        times = precip_raw.valid_time.values

        precip_in = conversion * precip_raw.values

        slr = slr_ds.interp(step=point_forecast.step).values[:-1]

        snow = np.zeros(precip_in.shape)
        precip_rate = np.diff(precip_in)
        snow_rate = precip_rate * slr
        snow[1:] = np.cumsum(snow_rate)

        axs[0, 0].plot(times, precip_in, label=model_name)
        axs[1, 0].plot(times, snow, label=model_name)

        all_precip.append(precip_in)
        all_snow.append(snow)

    for model_name, loader in ENSEMBLE_LOADERS.items():
        ds = loader.open_dataset()
        kdtree = loader.get_kdtree()
        point_forecast = utils.nearest_neighbor_forecast(ds, kdtree, lon, lat)

        for member in point_forecast.number.values:
            member_forecast = point_forecast.sel(number=member)
            precip_raw = member_forecast.tp
            conversion = utils.swe_to_in(precip_raw.units)
            times = precip_raw.valid_time.values

            precip_in = conversion * precip_raw.values

            slr = slr_ds.interp(step=member_forecast.step).values[:-1]

            snow = np.zeros(precip_in.shape)
            precip_rate = np.diff(precip_in)
            snow_rate = precip_rate * slr
            snow[1:] = np.cumsum(snow_rate)

            if member == 0:
                axs[0, 0].plot(
                    times, precip_in, label=model_name, color="deepskyblue", alpha=0.75
                )
                axs[1, 0].plot(
                    times, snow, label=model_name, color="deepskyblue", alpha=0.75
                )
            else:
                axs[0, 0].plot(times, precip_in, color="deepskyblue", alpha=0.75)
                axs[1, 0].plot(times, snow, color="deepskyblue", alpha=0.75)

            all_precip.append(precip_in)
            all_snow.append(snow)

    precip_mean = np.mean(all_precip, axis=0)
    snow_mean = np.mean(all_snow, axis=0)

    axs[0, 0].plot(
        times, precip_mean, linewidth=3, color="black", zorder=200, label="mean"
    )
    axs[1, 0].plot(
        times, snow_mean, linewidth=3, color="black", zorder=200, label="mean"
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
    slr_x = np.arange(1, len(slr) + 1)
    ax_slr.plot(slr_x, slr, color="gray", label="Snow Liquid Ratio")
    ax_slr.set_ylim((0, 30))
    ax_slr.legend()
    ax_slr.set_ylabel("Snow:Liquid Ratio")

    # turn second y axis labels on
    axs[0, 1].yaxis.set_tick_params(labelleft=True)
    axs[1, 1].yaxis.set_tick_params(labelleft=True)

    # Set xaxis labels for boxplots
    tick_interval = 2
    ticks = np.arange(0, len(times) + 1, tick_interval)
    axs[0, 1].set_xticks(ticks, labels=ticks)
    axs[1, 1].set_xticks(ticks, labels=ticks)

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


def download_all_forecasts(cycle=None, force=False):
    LOADERS = {
        "ARW": HireswArwLoader(),
        "ARW2": HireswArw2Loader(),
        "FV3": HireswFv3Loader(),
        "HRRR": HrrrLoader(),
        "RRFS": RrfsLoader(),
        "NBM": NbmLoader(short_term=True),
    }

    for name, loader in LOADERS.items():
        loader.download_forecast(cycle=cycle, force=force)
