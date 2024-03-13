import io
from datetime import datetime
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
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


def point_ensemble_to_df(
    da: xr.DataArray, ens_name: str = "ens", tz: str = "UTC"
) -> pd.DataFrame:
    """Converts ensemble point forcast to DataFrame.

    Column names are 'ens_name_{number}'.
    """
    times = localize_timestamps(da.valid_time.values, tz)
    data = {}
    for member_num in da.number:
        member_num = int(member_num)

        member = da.sel(number=member_num)
        data[f"{ens_name}_{member_num}"] = member.to_numpy()

    df = pd.DataFrame(index=times, data=data)
    return df


def drop_first_nans(df):
    """Processes the df to remove nans.

    Needs to recompute the total precip since
    some nonzero values might have been dropped.
    """
    df = df.diff()
    first_valid_index = df.dropna().index[0]
    df_cleaned = df.loc[first_valid_index:]
    df_cleaned = df_cleaned.cumsum()
    return df_cleaned


def create_point_forecast_dfs(
    lon: float, lat: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    LOADERS = {
        "ARW": HireswArwLoader(),
        "ARW2": HireswArw2Loader(),
        "FV3": HireswFv3Loader(),
        "HRRR": HrrrLoader(),
        "RRFS": RrfsLoader(),
    }

    local_tz = get_tz(lon, lat)
    nbm = NbmLoader(short_term=True)
    slr_ds = nbm.forecast_slr(lon, lat)
    slr_times = localize_timestamps(slr_ds.valid_time.values, local_tz)
    slr_df = pd.DataFrame(index=slr_times, data={"slr": slr_ds})

    # plt.figure()

    precip_dfs = []
    for model_name, loader in LOADERS.items():
        ds = loader.open_dataset()
        kdtree = loader.get_kdtree()
        point_forecast = utils.nearest_neighbor_forecast(ds, kdtree, lon, lat)
        precip_raw = point_forecast.tp
        conversion = utils.swe_to_in(precip_raw.units)
        precip_in_da = conversion * precip_raw

        if hasattr(precip_in_da, "number"):
            precip_df = point_ensemble_to_df(precip_in_da, model_name, local_tz)

        else:
            times = localize_timestamps(precip_raw.valid_time.values, local_tz)
            precip_df = pd.DataFrame(index=times, data={model_name: precip_in_da})

        precip_dfs.append(precip_df)

    precip_df = pd.concat(precip_dfs, axis=1)
    precip_df = drop_first_nans(precip_df)
    return precip_df, slr_df


def create_snow_df(precip_df: pd.DataFrame, slr_df: pd.DataFrame) -> pd.DataFrame:
    slr_df = slr_df.reindex(precip_df.index)
    slr_df = slr_df.resample("60min").interpolate()
    slr = slr_df.slr

    hourly_precip = precip_df.diff()
    hourly_snow = hourly_precip.mul(slr, axis=0)
    snow_df = hourly_snow.cumsum()
    return snow_df


def plume_plot(lon: float, lat: float, title="", return_bytes=False):

    local_tz = get_tz(lon, lat)
    ensembles = {"RRFS": "deepskyblue"}

    precip_df, slr_df = create_point_forecast_dfs(lon, lat)
    snow_df = create_snow_df(precip_df, slr_df)
    precip_rate_df = precip_df.diff()
    snow_rate_df = snow_df.diff()
    deterministic_models = [
        col
        for col in precip_df.columns
        if not any(ensemble_key in col for ensemble_key in ensembles.keys())
    ]

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"{title} lat: {lat} lon: {lon}")
    plt.tight_layout(
        pad=3,
    )

    total_precip_ax = axs[0, 0]
    total_snow_ax = axs[0, 1]
    hourly_precip_ax = axs[1, 0]
    hourly_snow_ax = axs[1, 1]


    for model_name in deterministic_models:
        times = precip_df.index
        total_precip_ax.plot(times, precip_df[model_name], label=model_name)
        total_snow_ax.plot(times, snow_df[model_name], label=model_name)

        # hourly
        hourly_precip_ax.plot(times, precip_rate_df[model_name], label=model_name)
        hourly_snow_ax.plot(times, snow_rate_df[model_name], label=model_name)

    for model_name, color in ensembles.items():
        alpha = 0.75

        ensemble_cols = [col for col in precip_df.columns if model_name in col]
        for member_name in ensemble_cols:
            times = precip_df.index
            total_precip_ax.plot(
                times,
                precip_df[member_name],
                label=model_name,
                color=color,
                alpha=alpha,
            )
            total_snow_ax.plot(
                times, snow_df[member_name], label=model_name, color=color, alpha=alpha
            )

            # hourly
            hourly_precip_ax.plot(
                times,
                precip_rate_df[member_name],
                label=model_name,
                color=color,
                alpha=alpha,
            )
            hourly_snow_ax.plot(
                times,
                snow_rate_df[member_name],
                label=model_name,
                color=color,
                alpha=alpha,
            )

    precip_mean = precip_df.mean(axis=1)
    snow_mean = snow_df.mean(axis=1)

    hourly_precip_mean = precip_rate_df.mean(axis=1)
    hourly_snow_mean = snow_rate_df.mean(axis=1)

    total_precip_ax.plot(
        times, precip_mean, linewidth=3, color="black", zorder=200, label="mean"
    )
    total_snow_ax.plot(
        times, snow_mean, linewidth=3, color="black", zorder=200, label="mean"
    )

    # hourly
    hourly_precip_ax.plot(
        times,
        hourly_precip_mean,
        linewidth=3,
        color="black",
        zorder=200,
        label="mean",
    )
    hourly_snow_ax.plot(
        times,
        hourly_snow_mean,
        linewidth=3,
        color="black",
        zorder=200,
        label="mean",
    )

    # Remove duplicate entries on legend
    handles, labels = total_precip_ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles)).items()
    unique_labels, unique_handles = zip(*unique)
    total_precip_ax.legend(unique_handles, unique_labels)
    hourly_precip_ax.legend(unique_handles, unique_labels)

    # SLR Line
    ax_slr = total_snow_ax.twinx()
    ax_slr.plot(slr_df.index, slr_df.slr, color="gray", label="Snow Liquid Ratio")
    ax_slr.set_ylim((0, 30))
    ax_slr.legend()
    ax_slr.set_ylabel("Snow:Liquid Ratio")

    ax_slr = hourly_snow_ax.twinx()
    # slr_times = [t.tz_convert(local_tz) for t in times]
    ax_slr.plot(slr_df.index, slr_df.slr, color="gray", label="Snow Liquid Ratio")

    ax_slr.set_ylim((0, 30))
    ax_slr.legend()
    ax_slr.set_ylabel("Snow:Liquid Ratio")

    # Grid dotted lines
    total_precip_ax.grid(axis="both", linestyle="--")
    hourly_precip_ax.grid(axis="both", linestyle="--")
    total_snow_ax.grid(axis="both", linestyle="--")
    hourly_snow_ax.grid(axis="both", linestyle="--")

    # Subplot titles
    total_precip_ax.title.set_text("Accumulated Precipitation")
    total_snow_ax.title.set_text("Accumulated Snow")
    hourly_precip_ax.title.set_text("Hourly Precip")
    hourly_snow_ax.title.set_text("Hourly Snow")

    # Subplot ylabels
    total_precip_ax.set_ylabel("Precip (in)")
    total_snow_ax.set_ylabel("Snow (in)")
    hourly_precip_ax.set_ylabel("Precip (in)")
    hourly_snow_ax.set_ylabel("Snow (in)")

    # Subplot xlabels
    hourly_precip_ax.set_xlabel(f"Time ({local_tz})")
    hourly_snow_ax.set_xlabel(f"Time ({local_tz})")

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
