from typing import Literal, Tuple
from datetime import datetime, timedelta, date
import requests
import os
import logging
from concurrent.futures import ThreadPoolExecutor

LOG_FORMAT = "%(asctime)s %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
logger = logging.getLogger("href")

Cycle = Literal["00z", "12z"]
Product = Literal["mean", "lpmm", "sprd"]


class ForecastHourString(str):
    def __new__(cls, value):
        if not isinstance(value, int):
            raise ValueError("Invalid input. Must be an integer.")

        if value < 0 or value > 99:
            raise ValueError("Invalid input. The integer must be between 0 and 99.")

        padded_value = f"{value:02}"
        return super(ForecastHourString, cls).__new__(cls, padded_value)


def make_fhour_str(fhour: int) -> str:
    return str(fhour).zfill(2)


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


def grib_filename(product: Product, cycle: Cycle, fhour: int):
    fhour_str = str(fhour).zfill(2)
    return f"href.t{cycle}z.conus.{product}.f{fhour_str}.grib2"


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
    params = {
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

    if resp.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        logger.error(
            f"Failed to download the {filename}. Status code: {resp.status_code}"
        )

    return resp.status_code


def test_download():
    day, cycle = latest_date_and_cycle(datetime.utcnow())

    fhours = range(1, 49)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                download_grib,
                day,
                cycle,
                "mean",
                fhour,
                left_lon=-110,
                right_lon=-100,
                top_lat=45,
                bottom_lat=35,
            )
            for fhour in fhours
        ]

        for future in futures:
            res = future.result()
            print(res)

    """
    download_grib(
        day, cycle, "mean", 1, left_lon=-110, right_lon=-100, top_lat=45, bottom_lat=35
    )
    """
