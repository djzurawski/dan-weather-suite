from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
import dan_weather_suite.plotting.regions as regions
import dask
from datetime import datetime, timezone
from dateutil.parser import isoparse
import logging
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import KDTree
import requests
from typing import Tuple
import xarray as xr
import bz2

DECOMPRESSORS = {"bz2": bz2.decompress}


logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


def midpoints(x: np.ndarray) -> np.ndarray:
    "Returs midpoints of array"
    return (x[1:] + x[:-1]) / 2


def parse_np_datetime64(t: np.datetime64) -> datetime:
    return isoparse(str(t)).replace(tzinfo=timezone.utc)


def np_timedelta64_to_hour(td: np.timedelta64) -> float:
    h = td.astype("timedelta64[ns]") / (3600 * 10**9)
    return h.astype(float)


def intepolate_to_fhour(fhours_interp, fhours, values):
    return np.interp(fhours_interp, fhours, values)


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
    # prevent slicing from creating large chunk and causing huge memory usage
    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        left = extent.left
        right = extent.right
        top = extent.top
        bottom = extent.bottom
        x_condition = (ds.longitude >= left).compute() & (
            ds.longitude <= right
        ).compute()
        y_condition = (ds.latitude >= bottom).compute() & (ds.latitude <= top).compute()
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


def download_and_combine_gribs(
    urls: list[Tuple[str, dict]], threads=2, compression: str | None = None
) -> bytes:
    with ThreadPoolExecutor(threads) as executor:

        results = list(executor.map(lambda x: download_bytes(*x), urls))

        if compression:
            if compression not in DECOMPRESSORS:
                raise ValueError(f"{compression} decompression not implemented")

            decompressor = DECOMPRESSORS[compression]
            results = [decompressor(res) for res in results]

        concatenated_bytes = b"".join(
            result for result in results if result is not None
        )
        return concatenated_bytes


def nearest_neighbor_forecast(
    ds: xr.Dataset, tree: KDTree, lon: float, lat: float
) -> xr.Dataset:
    k_nearest = 9
    pairs = np.dstack((ds.longitude, ds.latitude)).reshape(-1, 2)
    distances, indicies = tree.query([lon, lat], k_nearest)
    nearest_points = pairs[indicies]

    # (haversine dist, coord array indices)
    nearest_points_idx = [
        (haversine(lat, lon, nbm_lat, nbm_lon), idx)
        for (nbm_lon, nbm_lat), idx in zip(nearest_points, indicies)
    ]

    nearest_point_km, nearest_idx = min(nearest_points_idx, key=lambda x: x[0])
    logging.info(f"Nearest: {pairs[nearest_idx]} {round(nearest_point_km,2)}km")

    nearest_row, nearest_col = divmod(nearest_idx, ds.latitude.shape[1])
    return ds.sel(x=nearest_col, y=nearest_row)


def interp_forecast(
    da: xr.DataArray, tree: KDTree, lon: float, lat: float
) -> np.ndarray:
    "Only works on variable DataArray,"
    k_nearest = 9
    distances, indicies = tree.query([lon, lat], k_nearest)

    nearest_coords = [divmod(idx, da.latitude.shape[1]) for idx in indicies]
    lats = [da.sel(y=y, x=x).latitude.values for y, x in nearest_coords]
    lons = [da.sel(y=y, x=x).longitude.values for y, x in nearest_coords]
    data_points = [(lon, lat) for lon, lat in zip(lons, lats)]

    forecast = []
    for step in da.step:
        values = [da.sel(y=y, x=x, step=step).values for y, x in nearest_coords]
        interped = griddata(data_points, values, [(lon, lat)], method="linear")

        forecast.append(interped[0])
    return np.array(forecast)
