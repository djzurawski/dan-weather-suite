import argparse
import cartopy.crs as crs
import dan_weather_suite.plotting.plot as plot
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import gaussian_filter, uniform_filter, distance_transform_edt

import xarray as xr


def sum_data(data_dir) -> xr.Dataset:
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

    modified_ds = ds.isel(band=0).rename({"x": "longitude", "y": "latitude"})
    return modified_ds


def moving_average(ds, prism_resolution, model_resolution):
    data = ds.band_data
    scale_factor = round(model_resolution / prism_resolution)
    print("Scale factor:", scale_factor)
    smoothed = uniform_filter(data, (scale_factor, scale_factor), mode="nearest")
    return smoothed


def average_degree2(prism, latitudes, longitudes):
    latitudes = sorted(latitudes)
    prism_resolution_deg = np.abs(np.diff(prism.latitude))[0]
    model_resolution_deg = np.abs(np.diff(sorted(latitudes)))[0]
    ratio = int(round(model_resolution_deg / prism_resolution_deg))

    model_min_longitude = np.min(longitudes)
    model_min_latitude = np.min(latitudes)

    prism_min_longitude = np.min(prism.longitude.values)
    prism_min_latitude = np.min(prism.latitude.values)

    cells_up = round((model_min_latitude - prism_min_latitude) / prism_resolution_deg)
    cells_right = round(
        (model_min_longitude - prism_min_longitude) / prism_resolution_deg
    )

    print(cells_up, cells_right)

    averaged = xr.zeros_like(prism.band_data)

    n = 0
    for i in range(cells_right, len(prism.longitude), ratio):
        for j in range(cells_up, len(prism.latitude), ratio):
            top = j + ratio // 2
            bottom = j - ratio // 2
            left = i - ratio // 2
            right = i + ratio // 2

            cell = prism.isel(longitude=slice(left, right), latitude=slice(bottom, top))
            """
            if np.all(np.isnan(cell)):
                cell_mean = 0
            else:
                cell_mean = np.nanmean(cell.band_data)
            """
            cell_mean = np.nanmean(cell.band_data)
            averaged.isel(
                longitude=slice(left, right), latitude=slice(bottom, top)
            ).loc[:, :] = cell_mean

            n += 1
            if n % 1000 == 0:
                print(n)

    return averaged


def gaussian(prism_ds, sigma_deg):
    prism_resolution_deg = np.abs(np.diff(prism_ds.latitude))[0]
    sigma = round(sigma_deg / prism_resolution_deg)
    print(sigma, sigma_deg, prism_resolution_deg)
    return gaussian_filter(prism_ds.band_data, sigma=sigma, mode="mirror")


def fill_nan_with_nearest(arr: np.ndarray):
    nan_mask = np.isnan(arr)
    nearest_indices = distance_transform_edt(
        nan_mask, return_distances=False, return_indices=True
    )
    return arr[nearest_indices[0], nearest_indices[1]]


def main(data_dir, model_resolution_deg, outfile=None):
    prism = sum_data(data_dir)

    prism["band_data"] = xr.DataArray(
        fill_nan_with_nearest(prism.band_data.values), dims=["latitude", "longitude"]
    )

    print("summed data")
    sigma_deg = model_resolution_deg / 2
    smoothed = gaussian(prism, sigma_deg)
    print("smoothed data")

    output = prism.copy()
    output["smoothed"] = xr.DataArray(smoothed, dims=["latitude", "longitude"])

    ratio = prism.band_data / np.array(smoothed)
    output["ratio"] = xr.DataArray(ratio, dims=["latitude", "longitude"])
    output = output.rename({"band_data": "prism"})

    if outfile:
        output.to_netcdf(f"{outfile}.nc")
        print(f"saved to {outfile}.nc")

    return output


def plot_data(output):
    fig, ax = plot.create_basemap(display_counties=True)
    ax.pcolormesh(
        output.prism.longitude,
        output.prism.latitude,
        output.prism,
        transform=crs.PlateCarree(),
        vmax=600,
    )
    ax.set_title("prism")

    fig, ax = plot.create_basemap(display_counties=True)
    ax.pcolormesh(
        output.smoothed.longitude,
        output.smoothed.latitude,
        output.smoothed,
        transform=crs.PlateCarree(),
        vmax=600,
    )
    ax.set_title("smoothed")

    fig, ax = plot.create_basemap(display_counties=True)
    ax.pcolormesh(
        output.ratio.longitude,
        output.ratio.latitude,
        output.ratio,
        transform=crs.PlateCarree(),
        vmin=0.5,
        vmax=1.5,
    )

    ax.set_title("ratio")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory containing the prism .asc files",
    )

    parser.add_argument(
        "--resolution",
        type=float,
        help="Resolution of dataset to be downscaled in degrees",
    )

    parser.add_argument(
        "--name",
        type=str,
        help="Name of the output file",
    )
    args = parser.parse_args()

    main(args.data_dir, args.resolution, args.name)


def tst():
    output = main("/home/dan/Documents/prism", 0.5, "0.5deg-800m")
