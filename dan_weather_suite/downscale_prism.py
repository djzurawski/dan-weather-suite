import xarray as xr
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as crs
import dan_weather_suite.plotting.plot as plot
import os
import argparse


def sum_data(data_dir):
    """
    Sums all the precip data into one xr.Dataset
    """

    files = [f for f in os.listdir(data_dir) if ".asc" in f]

    ds = xr.open_dataset(os.path.join(data_dir, files[0]))

    for f in files[1:]:
        ds_f = xr.open_dataset(os.path.join(data_dir, f))
        total = ds.band_data
        da = xr.DataArray(ds_f.band_data + total, dims=["band", "y", "x"])
        ds["band_data"] = da

    return ds


def smooth_data(ds, prism_resolution, model_resolution):
    data = ds.band_data[0]
    scale_factor = round(model_resolution / prism_resolution)
    print("Smoothing factor:", scale_factor)
    kernel = np.ones((scale_factor, scale_factor)) / scale_factor**2

    # method='direct' takes forever but we need to use it to account for nan's
    # and we only need to do this once
    print("Smoothing data. This will take some time...")
    print(data.shape, kernel.shape)
    smoothed = signal.convolve(data, kernel, mode="same", method="direct")

    return smoothed


def main(data_dir, name):
    prism_resolution_m = 800
    model_resolution_m = 15 * 1000

    ds = sum_data(data_dir)
    print("summed data")

    smoothed = smooth_data(ds, prism_resolution_m, model_resolution_m)
    print("smoothed data")

    ratio = ds.band_data / np.array([smoothed])
    ratio = np.nan_to_num(ratio, nan=1)

    ds_ratio = ds.copy()
    ds_ratio["band_data"] = xr.DataArray(ratio, dims=["band", "y", "x"])

    ds_ratio.to_netcdf(f"{name}.nc")
    print(f"saved to {name}.nc")


def plot_data(lons, lats, data):
    fig, ax = plot.create_basemap()

    ax.pcolormesh(
        lons,
        lats,
        data,
        transform=crs.PlateCarree(),
    )

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory containing the prism .asc files",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Name of the output file",
    )
    args = parser.parse_args()

    main(args.data_dir, args.name)
