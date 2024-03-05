import io
from datetime import datetime
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tzfpy import get_tz

import dan_weather_suite.utils as utils
from dan_weather_suite.models.hiresw_arw import (
    HireswArw2Loader,
    HireswArwLoader,
    HireswFv3Loader,
)
from dan_weather_suite.models.hrrr import HrrrLoader
from dan_weather_suite.models.nbm import NbmLoader
from dan_weather_suite.models.rrfs import RrfsLoader


def localize_timestamps(
    times: Iterable[np.datetime64], local_tz: str
) -> list[datetime]:
    "Localizes vald_times from models"
    times = [pd.to_datetime(t).tz_localize("UTC").tz_convert(local_tz) for t in times]
    return times


def plume_plot(lon: float, lat: float, title="", return_bytes=False):
    LOADERS = {
        "ARW": HireswArwLoader(),
        "ARW2": HireswArw2Loader(),
        "FV3": HireswFv3Loader(),
        "HRRR": HrrrLoader(),
    }

    ENSEMBLE_LOADERS = {"RRFS": RrfsLoader()}

    local_tz = get_tz(lon, lat)

    nbm = NbmLoader(short_term=True)
    slr_ds = nbm.forecast_slr(lon, lat)

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
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
        times = localize_timestamps(precip_raw.valid_time.values, local_tz)

        precip_in = conversion * precip_raw.values

        slr_da = slr_ds.interp(step=point_forecast.step)
        slr = slr_da.values[:-1]

        snow = np.zeros(precip_in.shape)
        precip_rate = np.diff(precip_in)
        snow_rate = precip_rate * slr
        snow[1:] = np.cumsum(snow_rate)

        axs[0, 0].plot(times, precip_in, label=model_name)
        axs[0, 1].plot(times, snow, label=model_name)

        # hourly
        axs[1, 0].plot(times, np.gradient(precip_in), label=model_name)
        axs[1, 1].plot(times, np.gradient(snow), label=model_name)

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
            times = localize_timestamps(precip_raw.valid_time.values, local_tz)
            precip_in = conversion * precip_raw.values

            slr_da = slr_ds.interp(step=member_forecast.step)
            slr = slr_da.values[:-1]

            snow = np.zeros(precip_in.shape)
            precip_rate = np.diff(precip_in)
            snow_rate = precip_rate * slr
            snow[1:] = np.cumsum(snow_rate)

            if member == 0:
                axs[0, 0].plot(
                    times, precip_in, label=model_name, color="deepskyblue", alpha=0.75
                )
                # hourly
                axs[1, 0].plot(
                    times,
                    np.gradient(precip_in),
                    label=model_name,
                    color="deepskyblue",
                    alpha=0.75,
                )

                axs[0, 1].plot(
                    times, snow, label=model_name, color="deepskyblue", alpha=0.75
                )
                # hourly
                axs[1, 1].plot(
                    times,
                    np.gradient(snow),
                    label=model_name,
                    color="deepskyblue",
                    alpha=0.75,
                )
            else:
                axs[0, 0].plot(times, precip_in, color="deepskyblue", alpha=0.75)
                axs[0, 1].plot(times, snow, color="deepskyblue", alpha=0.75)

                # hourly
                axs[1, 0].plot(
                    times, np.gradient(precip_in), color="deepskyblue", alpha=0.75
                )
                axs[1, 1].plot(
                    times, np.gradient(snow), color="deepskyblue", alpha=0.75
                )

            all_precip.append(precip_in)
            all_snow.append(snow)

    precip_mean = np.mean(all_precip, axis=0)
    snow_mean = np.mean(all_snow, axis=0)

    axs[0, 0].plot(
        times, precip_mean, linewidth=3, color="black", zorder=200, label="mean"
    )
    axs[0, 1].plot(
        times, snow_mean, linewidth=3, color="black", zorder=200, label="mean"
    )

    # hourly
    axs[1, 0].plot(
        times,
        np.gradient(precip_mean),
        linewidth=3,
        color="black",
        zorder=200,
        label="mean",
    )
    axs[1, 1].plot(
        times,
        np.gradient(snow_mean),
        linewidth=3,
        color="black",
        zorder=200,
        label="mean",
    )

    axs[0, 0].legend()
    axs[1, 0].legend()

    # SLR Line
    slr_times = localize_timestamps(slr_da.valid_time.values, local_tz)
    ax_slr = axs[0, 1].twinx()
    ax_slr.plot(slr_times, slr_da, color="gray", label="Snow Liquid Ratio")
    ax_slr.set_ylim((0, 30))
    ax_slr.legend()
    ax_slr.set_ylabel("Snow:Liquid Ratio")

    ax_slr = axs[1, 1].twinx()
    # slr_times = [t.tz_convert(local_tz) for t in times]
    ax_slr.plot(slr_times, slr_da, color="gray", label="Snow Liquid Ratio")

    ax_slr.set_ylim((0, 30))
    ax_slr.legend()
    ax_slr.set_ylabel("Snow:Liquid Ratio")

    # Grid dotted lines
    axs[0, 0].grid(axis="both", linestyle="--")
    axs[1, 0].grid(axis="both", linestyle="--")
    axs[0, 1].grid(axis="both", linestyle="--")
    axs[1, 1].grid(axis="both", linestyle="--")

    # Subplot titles
    axs[0, 0].title.set_text("Accumulated Precipitation")
    axs[0, 1].title.set_text("Accumulated Snow")
    axs[1, 0].title.set_text("Hourly Precip")
    axs[1, 1].title.set_text("Hourly Snow")

    # Subplot ylabels
    axs[0, 0].set_ylabel("Precip (in)")
    axs[0, 1].set_ylabel("Snow (in)")
    axs[1, 0].set_ylabel("Precip (in)")
    axs[1, 1].set_ylabel("Snow (in)")

    # Subplot xlabels
    axs[1, 0].set_xlabel(f"Time ({local_tz})")
    axs[1, 1].set_xlabel(f"Time ({local_tz})")

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
