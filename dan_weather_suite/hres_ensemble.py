from dan_weather_suite.models.hiresw_arw import (
    HireswArwLoader,
    HireswArw2Loader,
    HireswFv3Loader,
)
from dan_weather_suite.models.hrrr import HrrrLoader
from dan_weather_suite.models.nbm import NbmLoader
from dan_weather_suite.models.rrfs import RrfsLoader
import dan_weather_suite.utils as utils
import io
import logging
import numpy as np
import matplotlib.pyplot as plt


# TODO handle not updated models
# TODO reduce nesting
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

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"{title} lat: {lat} lon: {lon}")
    plt.tight_layout(
        pad=3,
    )

    acc_precip_ax = axs[0, 0]
    acc_snow_ax = axs[0, 1]
    hr_precip_ax = axs[1, 0]
    hr_snow_ax = axs[1, 1]

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

        slr_da = slr_ds.interp(step=point_forecast.step)
        slr = slr_da.values[:-1]

        snow = np.zeros(precip_in.shape)
        precip_rate = np.diff(precip_in)
        snow_rate = precip_rate * slr
        snow[1:] = np.cumsum(snow_rate)

        acc_precip_ax.plot(times, precip_in, label=model_name)
        acc_snow_ax.plot(times, snow, label=model_name)

        # hourly
        hr_precip_ax.plot(times, np.gradient(precip_in), label=model_name)
        hr_snow_ax.plot(times, np.gradient(snow), label=model_name)

        all_precip.append(precip_in)
        all_snow.append(snow)

    for model_name, loader in ENSEMBLE_LOADERS.items():
        try:
            ds = loader.open_dataset()
            kdtree = loader.get_kdtree()
            point_forecast = utils.nearest_neighbor_forecast(ds, kdtree, lon, lat)

            for member in point_forecast.number.values:
                member_forecast = point_forecast.sel(number=member)
                precip_raw = member_forecast.tp
                conversion = utils.swe_to_in(precip_raw.units)
                times = precip_raw.valid_time.values

                precip_in = conversion * precip_raw.values

                slr_da = slr_ds.interp(step=member_forecast.step)
                slr = slr_da.values[:-1]

                snow = np.zeros(precip_in.shape)
                precip_rate = np.diff(precip_in)
                snow_rate = precip_rate * slr
                snow[1:] = np.cumsum(snow_rate)

                acc_precip_ax.plot(
                    times,
                    precip_in,
                    label=model_name,
                    color="deepskyblue",
                    alpha=0.75,
                )
                # hourly
                hr_precip_ax.plot(
                    times,
                    np.gradient(precip_in),
                    label=model_name,
                    color="deepskyblue",
                    alpha=0.75,
                )

                acc_snow_ax.plot(
                    times, snow, label=model_name, color="deepskyblue", alpha=0.75
                )
                # hourly
                hr_snow_ax.plot(
                    times,
                    np.gradient(snow),
                    label=model_name,
                    color="deepskyblue",
                    alpha=0.75,
                )

                all_precip.append(precip_in)
                all_snow.append(snow)

        except Exception as e:
            logging.error(f"Could lot plot {model_name}: {e}")

    precip_mean = np.mean(all_precip, axis=0)
    snow_mean = np.mean(all_snow, axis=0)

    acc_precip_ax.plot(
        times, precip_mean, linewidth=3, color="black", zorder=200, label="mean"
    )
    acc_snow_ax.plot(
        times, snow_mean, linewidth=3, color="black", zorder=200, label="mean"
    )

    # hourly
    hr_precip_ax.plot(
        times,
        np.gradient(precip_mean),
        linewidth=3,
        color="black",
        zorder=200,
        label="mean",
    )
    hr_snow_ax.plot(
        times,
        np.gradient(snow_mean),
        linewidth=3,
        color="black",
        zorder=200,
        label="mean",
    )

    # remove duplicate labels
    # https://stackoverflow.com/a/56253636
    handles, labels = acc_precip_ax.get_legend_handles_labels()
    unique = [
        (h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]
    ]
    acc_precip_ax.legend(*zip(*unique))
    hr_precip_ax.legend(*zip(*unique))

    # SLR Line
    ax_slr = acc_snow_ax.twinx()
    ax_slr.plot(
        slr_da.valid_time.values, slr_da, color="gray", label="Snow Liquid Ratio"
    )
    ax_slr.set_ylim((0, 30))
    ax_slr.legend()
    ax_slr.set_ylabel("Snow:Liquid Ratio")

    ax_slr = hr_snow_ax.twinx()
    ax_slr.plot(
        slr_da.valid_time.values, slr_da, color="gray", label="Snow Liquid Ratio"
    )
    ax_slr.set_ylim((0, 30))
    ax_slr.legend()
    ax_slr.set_ylabel("Snow:Liquid Ratio")

    # Grid dotted lines
    acc_precip_ax.grid(axis="both", linestyle="--")
    hr_precip_ax.grid(axis="both", linestyle="--")
    acc_snow_ax.grid(axis="both", linestyle="--")
    hr_snow_ax.grid(axis="both", linestyle="--")

    # Subplot titles
    acc_precip_ax.title.set_text("Accumulated Precipitation")
    acc_snow_ax.title.set_text("Accumulated Snow")
    hr_precip_ax.title.set_text("Hourly Precip")
    hr_snow_ax.title.set_text("Hourly Snow")

    # Subplot ylabels
    acc_precip_ax.set_ylabel("Precip (in)")
    acc_snow_ax.set_ylabel("Snow (in)")
    hr_precip_ax.set_ylabel("Precip (in)")
    hr_snow_ax.set_ylabel("Snow (in)")

    # Subplot xlabels
    hr_precip_ax.set_xlabel("Time (UTC)")
    hr_snow_ax.set_xlabel("Time (UTC)")

    if return_bytes:
        with io.BytesIO() as bio:
            plt.savefig(bio, format="jpg", bbox_inches="tight")
            plt.close(fig)
            return bio.getvalue()

    plt.show()


def download_all_forecasts(cycle=None, force=True):
    LOADERS = {
        "ARW": HireswArwLoader(),
        "ARW2": HireswArw2Loader(),
        "FV3": HireswFv3Loader(),
        "HRRR": HrrrLoader(),
        "RRFS": RrfsLoader(),
        "NBM": NbmLoader(short_term=True),
    }

    for name, loader in LOADERS.items():
        try:
            loader.download_forecast(cycle=cycle, force=force)
        except Exception as e:
            logging.error(f"Couldnt download {name} forecast:{e}")
