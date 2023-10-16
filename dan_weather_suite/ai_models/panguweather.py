from typing import Literal
from datetime import datetime
import xarray as xr
import numpy as np
import metpy
import metpy.calc as mpcalc
from metpy.units import units

from dan_weather_suite.plotting import plot
import matplotlib.pyplot as plt

from dan_weather_suite.plotting.regions import CONUS_EXTENT

import argparse
import os

test_date = datetime(2023, 10, 8, 12, 00, 00, 00)

Cycle = Literal["00z", "06z", "12z", "18z"]

NANOSECONDS_PER_HOUR = 60 * 60 * 10**9


def datetime64_to_datetime(dt: np.datetime64) -> datetime:
    return datetime.utcfromtimestamp(int(dt) / 1e9)


def load_grib(fpath: str, extent=None):
    ds = xr.open_dataset("panguweather.grib")
    ds = ds.metpy.parse_cf()
    ds["longitude"] = ((ds["longitude"] + 180) % 360) - 180
    if extent:
        ds = ds.sel(
            latitude=slice(extent.top, extent.bottom),
            longitude=slice(extent.left, extent.right),
        )

    return ds


def make_vort500_plots(ds, output_dir="images"):
    for step in ds.step:
        step_ds = ds.sel(step=step)
        init_dt = datetime64_to_datetime(step_ds.time)
        valid_dt = datetime64_to_datetime(step_ds.valid_time)

        fhour = int(step / NANOSECONDS_PER_HOUR)
        fhour_str = str(fhour).zfill(3)

        cycle = str(init_dt.hour).zfill(2)

        z_500 = step_ds.sel(isobaricInhPa=500).z / (9.81 * 10)
        u_500 = step_ds.sel(isobaricInhPa=500).u
        v_500 = step_ds.sel(isobaricInhPa=500).v
        vort_500 = (
            mpcalc.vorticity(u_500, v_500, latitude=ds.latitude, longitude=ds.longitude)
            * 10**5
        )

        fig, ax = plot.plot_500_vorticity(
            ds.longitude,
            ds.latitude,
            z_500,
            vort_500,
            u_500,
            v_500,
            barb_density=35,
            display_counties=False,
        )

        title = plot.make_title_str(
            init_dt,
            valid_dt,
            fhour,
            "500mb vorticity",
            "Pangu (GFS Init)",
            "10^5 s^-1",
            168,
        )
        ax.set_title(title)

        fname = f"{output_dir}/pangu.{cycle}z.conus.vort500.f{fhour_str}.png"
        print(init_dt.isoformat(), fname)

        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)


def make_rh700_plots(ds, output_dir="images"):
    for step in ds.step:
        step_ds = ds.sel(step=step)
        init_dt = datetime64_to_datetime(step_ds.time)
        valid_dt = datetime64_to_datetime(step_ds.valid_time)

        fhour = int(step / NANOSECONDS_PER_HOUR)
        fhour_str = str(fhour).zfill(3)

        cycle = str(init_dt.hour).zfill(2)

        z_700 = step_ds.sel(isobaricInhPa=700).z / (9.81 * 10)
        u_700 = step_ds.sel(isobaricInhPa=700).u
        v_700 = step_ds.sel(isobaricInhPa=700).v
        t_700 = step_ds.sel(isobaricInhPa=700).t
        q_700 = step_ds.sel(isobaricInhPa=700).q

        rh_700 = mpcalc.relative_humidity_from_specific_humidity(
            700 * units.hPa, t_700, q_700
        )

        rh_700 = np.clip(rh_700 * 100, 0, 100)

        fig, ax = plot.plot_700_rh(
            ds.longitude,
            ds.latitude,
            z_700,
            rh_700,
            u_700,
            v_700,
            barb_density=35,
            display_counties=False,
        )

        title = plot.make_title_str(
            init_dt,
            valid_dt,
            fhour,
            "700mb Relative Humidity",
            "Pangu (GFS Init)",
            "%",
            168,
        )
        ax.set_title(title)

        fname = f"{output_dir}/pangu.{cycle}z.conus.rh700.f{fhour_str}.png"
        print(init_dt.isoformat(), fname)

        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output-dir", help="Ouput directory for plots", default="images"
    )
    args = parser.parse_args()

    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ds = load_grib("panguweather.grib", CONUS_EXTENT)
    make_vort500_plots(ds, output_dir=output_dir)
    make_rh700_plots(ds, output_dir=output_dir)
