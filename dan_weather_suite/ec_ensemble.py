from concurrent.futures import ThreadPoolExecutor
from dan_weather_suite.plotting.regions import CONUS_EXTENT
from datetime import datetime, timedelta, date
import logging
import json
import numpy as np
import os
import requests
from scipy.interpolate import RegularGridInterpolator
import xarray as xr

LOG_FORMAT = "%(levelname)s %(asctime)s %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
logger = logging.getLogger("ec-ens")


def set_ds_extent(ds, extent=CONUS_EXTENT):
    # left, right, bottom, top = extent
    x_condition = (ds.longitude >= extent.left) & (ds.longitude <= extent.right)
    y_condition = (ds.latitude >= extent.bottom) & (ds.latitude <= extent.top)
    trimmed = ds.where(x_condition & y_condition, drop=True)
    return trimmed


def resample_to_grid(
    data: np.ndarray,
    data_lons: np.ndarray,
    data_lats: np.ndarray,
    grid_lons: np.ndarray,
    grid_lats: np.ndarray,
):
    interp_x, interp_y = np.meshgrid(grid_lons, grid_lats)
    interp_points = np.dstack([interp_y.ravel(), interp_x.ravel()])[0]
    interpolator = RegularGridInterpolator((data_lats, data_lons), data)
    results = interpolator(interp_points)
    output_shape = (grid_lats.shape[0], grid_lons.shape[0])
    return results.reshape(output_shape)


def field_byte_range(field_index: dict) -> str:
    start_bytes = field_index["_offset"]
    length = field_index["_length"]
    end_bytes = start_bytes + length

    return f"{start_bytes}-{end_bytes}"


def format_byte_header(byte_ranges: list[str]) -> str:
    byte_part = ", ".join(byte_ranges)
    return f"bytes={byte_part}"


def download_grib_field(day: date, cycle: int, fhour: int, field: str = "tp"):
    save_directory = "grib"

    day_str = day.strftime("%Y%m%d")

    root = "https://data.ecmwf.int/forecasts/"
    prefix = f"{day_str}/{cycle}z/0p4-beta/enfo/"
    filename = f"{day_str}{cycle}0000-{fhour}h-enfo-ef"
    fhour_url = f"{root}" + prefix + filename

    padded_fhour = str(fhour).zfill(3)
    padded_filename = f"{day_str}{cycle}0000-{padded_fhour}h-enfo-ef"

    index_resp = requests.get(fhour_url + ".index")
    lines = index_resp.text.split("\n")

    grib_index = []
    for line in lines:
        if len(line) > 0:
            grib_index.append(json.loads(line))

    param_index = [i for i in grib_index if i["param"] == field]
    field_bytes = [field_byte_range(i) for i in param_index]

    byte_header = {"Range": format_byte_header(field_bytes)}

    logging.info(f"Downloading: {filename}")
    data_resp = requests.get(fhour_url + ".grib2", headers=byte_header)

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    save_path = os.path.join(save_directory, padded_filename + ".grib2")
    with open(save_path, "wb") as f:
        f.write(data_resp.content)


def download_forecast(day: date, cycle: int, flength: int, field="tp"):
    step = 6

    fhours = range(step, flength + step, step)

    with ThreadPoolExecutor(24) as executor:
        futures = [
            executor.submit(download_grib_field, day, cycle, fhour, "tp")
            for fhour in fhours
        ]
        [f.result() for f in futures]


def test_download():
    day = day = date(2023, 11, 13)
    cycle = 12

    # download_precip_gribs(day, cycle, 24)
    download_forecast(day, cycle, 168)


def open_entire_forecast():
    # control member
    ds_c = xr.open_mfdataset(
        "*grib/*enfo-ef*.grib2",
        engine="cfgrib",
        combine="nested",
        concat_dim="time",  # or 'step'
        coords="minimal",
        data_vars="minimal",
        compat="override",
        filter_by_keys={"dataType": "cf"},
    )

    # perturbed members
    ds_p = xr.open_mfdataset(
        "*grib/*enfo-ef*.grib2",
        engine="cfgrib",
        combine="nested",
        concat_dim="time",  # or 'step'
        coords="minimal",
        data_vars="minimal",
        compat="override",
        filter_by_keys={"dataType": "pf"},
    )

    # add on the control member to the perturbed
    ds_c_expanded = ds_c.expand_dims("number", axis=1)

    ds = xr.concat([ds_c_expanded, ds_p], "number")

    return ds
