from typing import Literal, Tuple, Dict, Sequence
from datetime import datetime, timedelta, date
import requests
import os
import logging
from concurrent.futures import ThreadPoolExecutor
import subprocess

from dan_weather_suite.plotting.regions import Region, Extent, HREF_REGIONS

import haversine

import xarray as xr


import cfgrib
from dan_weather_suite.plotting import plot
import numpy as np

import matplotlib.pyplot as plt

LOG_FORMAT = "%(levelname)s %(asctime)s %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
logger = logging.getLogger("href")

Cycle = Literal["00z", "12z"]
Product = Literal["mean", "lpmm", "sprd"]
UrlParams = Dict[str, str | int | float]

FORECAST_LENGTH = 48  # hours

ALL_PRODUCTS: Sequence[Product] = ["mean", "lpmm", "sprd"]

MM_PER_IN = 25.4


def datetime64_to_datetime(dt: np.datetime64) -> datetime:
    return datetime.utcfromtimestamp(int(dt) / 1e9)


def select_cycle(dt: datetime) -> Cycle:
    """Empirically figured out the most recent available HREF cycle"""
    utc_hour = dt.hour
    if utc_hour > 15 or utc_hour < 3:
        return "12z"
    else:
        return "00z"


def select_day(dt: datetime) -> date:
    """Selects day of most recent HREF run"""

    hour = dt.hour
    if hour < 3:
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
        "subregion": "",
        "leftlon": left_lon,
        "rightlon": right_lon,
        "toplat": top_lat,
        "bottomlat": bottom_lat,
        "dir": f"/href.{day_str}/ensprod",
    }

    resp = requests.get(
        grib_filter_base_url, params=params, stream=True, allow_redirects=True
    )

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    save_path = os.path.join(save_directory, filename)

    if resp.status_code == 200:
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


def download_forecast(day: date, cycle: Cycle, product: Product, extent: Extent):
    top_lat = extent.top
    bottom_lat = extent.bottom
    left_lon = extent.left
    right_lon = extent.right

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


def max_extent(extents: Sequence[Extent]) -> Extent:
    top = max([e.top for e in extents])
    bottom = min([e.bottom for e in extents])
    left = min([e.left for e in extents])
    right = max([e.right for e in extents])

    return Extent(top=top, bottom=bottom, left=left, right=right)


def download_gribs(day: date, cycle: Cycle, product: Product, extent: Extent):
    download_forecast(day, cycle, product, extent)
    combine_grib_files(cycle, product)


def combined_gribfile(cycle: Cycle, product: Product) -> str:
    return f"grib/href.t{cycle}.conus.{product}_combined.grib2"


def cumsum_precip_in(ds):
    "Accumulates precip over fhours and converts mm precip to in"
    return np.cumsum(ds.tp, axis=0) / MM_PER_IN


def make_surface_plots(cycle: Cycle, product: Product, regions: Sequence[Region]):
    ds = cfgrib.open_dataset(combined_gribfile(cycle, product))

    cum_precip_in = cumsum_precip_in(ds)
    for i, forecast in enumerate(cum_precip_in):
        fhour = i + 1
        fhour_str = str(fhour).zfill(2)
        init_time = datetime64_to_datetime(forecast.time)
        valid_time = datetime64_to_datetime(forecast.valid_time)

        plot.make_title_str(init_time, valid_time, fhour, product, "HREF", "in")

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

        for region in regions:
            ax.set_extent(
                [
                    region.extent.left,
                    region.extent.right,
                    region.extent.bottom,
                    region.extent.top,
                ]
            )
            fig, ax = plot.add_labels(fig, ax, region.labels)
            fname = f"href.{cycle}.{region.name}.{product}.f{fhour_str}.png"
            logging.info(f"Saving {fname}")
            fig.savefig(f"images/{fname}")

        plt.close(fig)


def nearest_point(ds: xr.Dataset, lon: float, lat: float):
    """Finds the nearest lat/lon and (x,y) index for given lat/lon
    in xarray.Dataset"""

    def haver(ds_lat: float, ds_lon: float):
        if ds_lon > 180 or ds_lon < 0:
            ds_lon = ds_lon - 360
        return haversine.haversine((lat, lon), (ds_lat, ds_lon))

    vectorized_haver = np.vectorize(haver)

    distances = vectorized_haver(ds.latitude, ds.longitude)
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmin.html
    #  - example with np.unravel_index
    lat_idx, lon_idx = np.unravel_index(
        np.argmin(distances, axis=None), distances.shape
    )
    nearest = ds.sel(x=lon_idx, y=lat_idx)
    nearest_lat = float(nearest.latitude.values)
    nearest_lon = float(nearest.longitude.values)
    if nearest_lon > 180 or nearest_lon < 0:
        nearest_lon = nearest_lon - 360

    return (lon_idx, lat_idx), (nearest_lon, nearest_lat)


def make_point_plots(cycle: Cycle, regions: Sequence[Region]):
    for region in regions:
        for label in region.labels:
            plt.figure(figsize=(12, 7))
            for product in ALL_PRODUCTS:
                ds = cfgrib.open_dataset(combined_gribfile(cycle, product))

                initialized = np.datetime_as_string(ds.time, unit="m", timezone="UTC")

                ds_cum = cumsum_precip_in(ds)
                ((x_idx, y_idx), (nearest_lon, nearest_lat)) = nearest_point(
                    ds, label.lon, label.lat
                )

                location_title = "{} ({}, {})".format(
                    label.text, round(nearest_lat, 3), round(nearest_lon, 3)
                )

                ds_loc = ds_cum.sel(x=x_idx, y=y_idx)

                x = [(ds_loc.time + step).values for step in ds_loc.step]
                y = ds_loc
                plt.plot(x, y, label=product)

            plt.title("HREF Initialized: " + initialized, loc="left", fontsize=18)
            plt.title(location_title, loc="right", fontsize=18)

            plt.legend()
            fname = f"{label.text}.t{cycle}.png"
            logging.info(f"Saving {fname}")
            plt.savefig(f"images/{fname}")
            plt.close()


def main():
    day, cycle = latest_date_and_cycle(datetime.utcnow())

    download_extent = max_extent([r.extent for r in HREF_REGIONS])

    for product in ALL_PRODUCTS:
        download_gribs(day, cycle, product, download_extent)

    for product in ALL_PRODUCTS:
        make_surface_plots(cycle, product, HREF_REGIONS)

    make_point_plots(cycle, HREF_REGIONS)


if __name__ == "__main__":
    main()
