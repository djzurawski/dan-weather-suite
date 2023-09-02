from typing import Literal, Tuple, Dict
from datetime import datetime, timedelta, date
import requests
import os
import logging
from concurrent.futures import ThreadPoolExecutor
import subprocess

from dan_weather_suite.plotting.regions.models import Region
from dan_weather_suite.plotting.regions.href import FRONT_RANGE

import cfgrib
from dan_weather_suite.plotting import plot
import numpy as np

import matplotlib.pyplot as plt

LOG_FORMAT = "%(asctime)s %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
logger = logging.getLogger("href")

Cycle = Literal["00z", "12z"]
Product = Literal["mean", "lpmm", "sprd"]
UrlParams = Dict[str, str | int | float]

FORECAST_LENGTH = 48  # hours

PRODUCTS = ["mean", "lpmm", "sprd"]


def select_cycle(dt: datetime) -> Cycle:
    """Empirically figured out the most recent available HREF cycle"""
    utc_hour = dt.hour
    if utc_hour > 15 or utc_hour < 4:
        return "12z"
    else:
        return "00z"


def select_day(dt: datetime) -> date:
    """Selects day of most recent HREF run"""

    hour = dt.hour
    if hour < 4:
        return dt.date() - timedelta(days=1)
    else:
        return dt.date()


def latest_date_and_cycle(dt: datetime) -> Tuple[date, Cycle]:
    """Finds the most recent HREF init day and cycle"""

    cycle = select_cycle(dt)
    day = select_day(dt)

    return day, cycle


def grib_filename(product: Product, cycle: Cycle, fhour: int) -> str:
    fhour_str = str(fhour).zfill(2)
    return f"href.t{cycle}.conus.{product}.f{fhour_str}.grib2"


def download_grib(
    day: date,
    cycle: Cycle,
    product: Product,
    fhour: int,
    left_lon: float,
    right_lon: float,
    top_lat: float,
    bottom_lat: float,
):
    fhour_str = str(fhour).zfill(2)
    day_str = day.strftime("%Y%m%d")

    save_directory = "grib"
    filename = grib_filename(product, cycle, fhour)

    logger.info(f"Downloading {filename}")

    grib_filter_base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrefconus.pl"
    params: UrlParams = {
        "file": f"href.t{cycle}.conus.{product}.f{fhour_str}.grib2",
        "lev_surface": "on",
        "leftlon": left_lon,
        "rightlon": right_lon,
        "toplat": top_lat,
        "bottomlat": bottom_lat,
        "dir": f"/href.{day_str}/ensprod",
    }

    resp = requests.get(grib_filter_base_url, params=params, stream=True)

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    save_path = os.path.join(save_directory, filename)

    if resp.status_code == 200 or resp.status_code == 302:
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        logger.error(
            f"Failed to download the {filename}. Status code: {resp.status_code}"
        )

    return resp.status_code


def combine_grib_files(cycle, product, grib_dir="grib"):
    """'cat's all the hourly forecast gribs for 'cycle' and 'product into one
    big grib"""

    files_to_combine = []

    for input_file in sorted(os.listdir(grib_dir)):
        if f"href.t{cycle}.conus.{product}.f" in input_file:
            files_to_combine.append(os.path.join(grib_dir, input_file))

    if files_to_combine:
        output_file = os.path.join(
            grib_dir, f"href.t{cycle}.conus.{product}_combined.grib2"
        )
        cat_command = f'cat {" ".join(files_to_combine)} > {output_file}'

        subprocess.call(cat_command, shell=True)


def download_forecast(day: date, cycle: Cycle, product: Product, region: Region):
    top_lat = region.extent.top
    bottom_lat = region.extent.bottom
    left_lon = region.extent.left
    right_lon = region.extent.right

    fhours = range(1, FORECAST_LENGTH + 1)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                download_grib,
                day,
                cycle,
                product,
                fhour,
                left_lon=left_lon,
                right_lon=right_lon,
                top_lat=top_lat,
                bottom_lat=bottom_lat,
            )
            for fhour in fhours
        ]

        [f.result() for f in futures]


def download_gribs(day, cycle, product):
    for product in PRODUCTS:
        download_forecast(day, cycle, product, FRONT_RANGE)
        combine_grib_files(cycle, product)


def make_plots(cycle):
    regions = [FRONT_RANGE]
    for product in PRODUCTS:
        ds = cfgrib.open_dataset(f"grib/href.t{cycle}.conus.{product}_combined.grib2")

        MM_PER_IN = 25.4

        cum_precip_in = np.cumsum(ds.tp, axis=0) / MM_PER_IN
        for i, forecast in enumerate(cum_precip_in):
            fhour = i + 1
            fhour_str = str(fhour).zfill(2)
            print(fhour)
            init_time = forecast.time
            valid_time = forecast.valid_time

            fig, ax = plot.create_basemap()
            fig, ax = plot.add_contourf(
                fig,
                ax,
                ds.longitude,
                ds.latitude,
                forecast,
                levels=plot.PRECIP_CLEVS,
                colors=plot.PRECIP_CMAP_DATA,
            )

            fig.savefig(f"href.{cycle}.conus.{product}.f{fhour_str}.png")

            for region in regions:
                ax.set_extent(
                    [
                        FRONT_RANGE.extent.left,
                        FRONT_RANGE.extent.right,
                        FRONT_RANGE.extent.bottom,
                        FRONT_RANGE.extent.top,
                    ]
                )
                fig.savefig(f"href.{cycle}.{region.name}.{product}.f{fhour_str}.png")

            plt.close(fig)


def main():
    day, cycle = latest_date_and_cycle(datetime.utcnow())
    cycle = "12z"


def tst():
    f = "/home/dan/Sourcecode/dan-weather-suite/grib/href.t12z.conus.mean_combined.grib2"
    ds = cfgrib.open_dataset(f)

    dsc = np.cumsum(ds.tp, axis=0)

    fig, ax = plot.create_basemap()

    fig, ax = plot.add_contourf(
        fig,
        ax,
        ds.longitude,
        ds.latitude,
        dsc[-1],
    )
    fig.savefig("test.png")

    ax.set_extent(
        [
            FRONT_RANGE.extent.left,
            FRONT_RANGE.extent.right,
            FRONT_RANGE.extent.bottom,
            FRONT_RANGE.extent.top,
        ]
    )

    fig.savefig("test2.png")
