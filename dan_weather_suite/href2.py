from dan_weather_suite.models.hiresw_arw import (
    HireswArwLoader,
    HireswArw2Loader,
    HireswFv3Loader,
)
from dan_weather_suite.models.hrrr import HrrrLoader
from dan_weather_suite.models.nbm import NbmLoader
import dan_weather_suite.utils as utils
import numpy as np
import matplotlib.pyplot as plt


def plume_plot(lon:float, lat:float, title=""):
    LOADERS = {
        "ARW": HireswArwLoader(),
        "ARW2": HireswArw2Loader(),
        "FV3": HireswFv3Loader(),
        "HRRR": HrrrLoader(),
    }

    nbm = NbmLoader()
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
        snow_rate = precip_rate * 15
        snow[1:] = np.cumsum(snow_rate)

        axs[0, 0].plot(times, precip_in, label=model_name)
        axs[1, 0].plot(times, snow, label=model_name)

        all_precip.append(precip_in)
        all_snow.append(snow)

    precip_mean = np.mean(all_precip, axis=0)
    snow_mean = np.mean(all_snow, axis=0)

    axs[0, 0].plot(times, precip_mean, linewidth=3,color='black', zorder=200, label="mean")
    axs[1, 0].plot(times, snow_mean, linewidth=3, color='black', zorder=200, label="mean")

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

    # turn second y axis labels on
    axs[0, 1].yaxis.set_tick_params(labelleft=True)
    axs[1, 1].yaxis.set_tick_params(labelleft=True)

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

    plt.show()


def download_all_forecasts(cycle=None, force=False):
    LOADERS = {
        "ARW": HireswArwLoader(),
        "ARW2": HireswArw2Loader(),
        "FV3": HireswFv3Loader(),
        "HRRR": HrrrLoader(),
        "NBM": NbmLoader(),
    }

    for name, loader in LOADERS.items():
        loader.download_forecast(cycle=cycle, force=force)
