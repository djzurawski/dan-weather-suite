from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from dateutil.parser import isoparse
import numpy as np
import dan_weather_suite.plotting.regions as regions
import logging
import requests
from typing import Tuple
import xarray as xr


logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


def parse_np_datetime64(t: np.datetime64) -> datetime:
    return isoparse(str(t)).replace(tzinfo=timezone.utc)


def round_to_nearest(x: float, options: Iterable[float] = (0.25, 0.4, 0.5)) -> float:
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
        logging.info(f"downloading {url} {params}")
        resp = requests.get(url, params=params, timeout=600)
        if resp.status_code == 200:
            result = resp.content
            return result
        else:
            error_str = (
                f"Error downloading {url} status:{resp.status_code}, {resp.text}"
            )
            raise requests.exceptions.RequestException(error_str)

    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading {url} status:{resp.status_code}, {e}")
        return None


def download_and_combine_gribs(urls: list[Tuple[str, dict]], threads=2) -> bytes:
    with ThreadPoolExecutor(threads) as executor:
        results = list(executor.map(lambda x: download_bytes(*x), urls))
        concatenated_bytes = b"".join(
            result for result in results if result is not None
        )
        return concatenated_bytes
