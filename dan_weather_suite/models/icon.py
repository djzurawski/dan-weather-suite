import xarray as xr
from scipy.interpolate import SmoothBivariateSpline, LinearNDInterpolator
import numpy as np
from dan_weather_suite.plotting import plot
from dan_weather_suite import utils
import simplekml
from scipy.spatial import KDTree

from dan_weather_suite.models.loader import ModelLoader
import dan_weather_suite.utils as utils
from datetime import datetime, time, timedelta
from typing import Tuple
import xarray as xr
import os
import bz2


class IconLoader(ModelLoader):
    def __init__(self):
        super().__init__()
        self.forecast_length = 180
        forecast_hours_6 = list(range(0, 126, 6))
        forecast_hours_12 = list(range(132, 180 + 12, 12))
        self.forecast_hours = forecast_hours_6 + forecast_hours_12
        self.grib_file = "grib/icon.grib"
        self.netcdf_file = "grib/icon.nc"

        self.lons_path = (
            "grib/icon-eps_global_icosahedral_time" "-invariant_2024032400_clon.grib2"
        )
        self.lats_path = (
            "grib/icon-eps_global_icosahedral_time" "-invariant_2024032400_clat.grib2"
        )

    def get_latest_init(self) -> datetime:
        current_utc = datetime.utcnow()
        current_utc_time = current_utc.time()

        release_00z = time(3, 30)
        release_12z = time(15, 30)

        if release_00z <= current_utc_time < release_12z:
            return datetime(current_utc.year, current_utc.month, current_utc.day, 0, 0)
        elif current_utc_time >= release_12z:
            return datetime(current_utc.year, current_utc.month, current_utc.day, 12, 0)
        else:
            previous_day = current_utc - timedelta(days=1)
            return datetime(
                previous_day.year, previous_day.month, previous_day.day, 12, 0
            )

    def download_grib(self, cycle=None):
        init_dt = self.get_latest_init()
        if cycle is not None:
            init_dt = init_dt.replace(hour=cycle)

        urls = [self.url_formatter(init_dt, fhour) for fhour in self.forecast_hours]

        grib_bytes = utils.download_and_combine_gribs(
            urls, compression="bz2", threads=4
        )

        with open(self.grib_file, "wb") as f:
            f.write(grib_bytes)

    def url_formatter(self, init_dt: datetime, fhour) -> Tuple[str, dict]:
        day_str = init_dt.strftime("%Y%m%d%H")
        cycle_str = str(init_dt.hour).zfill(2)
        fhour_str = str(fhour).zfill(3)

        root_url = "https://opendata.dwd.de/weather/nwp/icon-eps/grib"

        return (
            (
                f"{root_url}/{cycle_str}/tot_prec/"
                f"icon-eps_global_icosahedral_single-level_{day_str}_"
                f"{fhour_str}_tot_prec.grib2.bz2"
            ),
            {},
        )

    def download_coordinates(self):

        lons_url = (
            "https://opendata.dwd.de/weather/nwp/icon-eps/"
            "grib/00/clon/icon-eps_global_icosahedral_time"
            "-invariant_2024032400_clon.grib2.bz2"
        )
        lats_url = (
            "https://opendata.dwd.de/weather/nwp/icon-eps"
            "/grib/00/clat/icon-eps_global_icosahedral_time"
            "-invariant_2024032400_clat.grib2.bz2"
        )

        if not os.path.exists(self.lons_path):
            grib_bytes = bz2.decompress(utils.download_bytes(lons_url))
            with open(self.lons_path, "wb") as f:
                f.write(grib_bytes)

        if not os.path.exists(self.lats_path):
            grib_bytes = bz2.decompress(utils.download_bytes(lats_url))
            with open(self.lats_path, "wb") as f:
                f.write(grib_bytes)

    def process_grib(self) -> xr.Dataset:

        self.download_coordinates()
        ds_lats = xr.open_dataset(self.lats_path)
        ds_lons = xr.open_dataset(self.lons_path)
        lats = ds_lats.tlat
        lons = ds_lons.tlon
        ds = xr.open_dataset(self.grib_file, chunks = {})
        ds = ds.swap_dims({"step": "valid_time"})

        lon_filter = (lons > -125) & (lons < -60)
        lat_filter = (lats > 25) & (lats < 60)

        coord_filter = lon_filter & lat_filter

        lons = lons[coord_filter].values
        lats = lats[coord_filter].values

        native_coords = np.column_stack((lons, lats))

        grid_lats = np.arange(-180, 180, 0.125)
        grid_lons = np.arange(-90, 90, 0.125)
        # our grid to interpolate to
        grid_lons, grid_lats = np.meshgrid(grid_lons, grid_lats)

        import time

        data_arrays = []
        for number in ds.number.values:
            member_forecast = []
            t0 = time.time()
            for valid_time in ds.valid_time.values:

                z = ds.sel(number=number, valid_time=valid_time).tp.values[coord_filter]
                print(z.shape)
                interpolator = LinearNDInterpolator(native_coords, z)
                interped = interpolator(grid_lons, grid_lats)
                member_forecast.append(interped)
            print(time.time() - t0)

            member_da = xr.DataArray(data=member_forecast,
                                     dims=["valid_time", "latitude", "longitude"],
                                     coords = {"valid_time": ds.valid_time.values,
                                               "latitude": grid_lats[:, 0],
                                               "longitude": grid_lons[0, :]})
            data_arrays.append(member_da)

        combined_da = xr.concat(data_arrays, dim='number')

        return combined_da


def dist2():
    df = xr.open_dataset(
        "/home/dan/Downloads/icon-eps_global_icosahedral_single-level_2024032312_168_tot_prec.grib2"
    )

    df = xr.open_dataset(
        "/home/dan/Downloads/icon-eps_global_icosahedral_time-invariant_2024032312_elat.grib2"
    )

    df_lats = xr.open_dataset(
        "/home/dan/Downloads/icon-eps_global_icosahedral_time-invariant_2024032312_clat.grib2"
    )
    lats = df_lats.tlat

    df_lons = xr.open_dataset(
        "/home/dan/Downloads/icon-eps_global_icosahedral_time-invariant_2024032312_clon.grib2"
    )
    lons = df_lons.tlon

    lon_filter = (lons > -117) & (lons < -100)
    lat_filter = (lats > 31) & (lats < 45)

    coord_filter = lon_filter & lat_filter

    lons = lons[coord_filter].values
    lats = lats[coord_filter].values

    lon0 = lons[0]
    lat0 = lats[0]

    distances = []

    points = np.array(list(zip(lons, lats)))
    tree = KDTree(points)

    for point in points:
        dists, idxs = tree.query(point, 4)

        closest = points[idxs]
        point_dists = []
        for lon1, lat1 in closest:
            point_dists.append(utils.haversine(points[1], points[0], lat1, lon1))
        avg = np.mean(point_dists)
        distances.append(avg)

    return distances


def google_earth():
    df = xr.open_dataset(
        "/home/dan/Downloads/icon-eps_global_icosahedral_single-level_2024032312_168_tot_prec.grib2"
    )

    df = xr.open_dataset(
        "/home/dan/Downloads/icon-eps_global_icosahedral_time-invariant_2024032312_elat.grib2"
    )

    df_lats = xr.open_dataset(
        "/home/dan/Downloads/icon-eps_global_icosahedral_time-invariant_2024032312_clat.grib2"
    )
    lats = df_lats.tlat

    df_lons = xr.open_dataset(
        "/home/dan/Downloads/icon-eps_global_icosahedral_time-invariant_2024032312_clon.grib2"
    )
    lons = df_lons.tlon

    lon_filter = (lons > -117) & (lons < -100)
    lat_filter = (lats > 31) & (lats < 45)

    coord_filter = lon_filter & lat_filter

    lons2 = lons[coord_filter].values
    lats2 = lats[coord_filter].values

    kml = simplekml.Kml()

    # Add points to the KML
    for lat, lon in zip(lats2, lons2):
        point = kml.newpoint(
            name="Point", coords=[(lon, lat)]
        )  # Note: KML expects lon, lat order
        # point.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'  # Optional: set a placemark icon

    # Save the KML file
    kml.save("my_points.kml")


def grid():
    import time
    df = xr.open_dataset(
        "/home/dan/Downloads/icon-eps_global_icosahedral_single-level_2024032312_168_tot_prec.grib2"
    )
    #df = xr.open_dataset("grib/icon-raw.nc").isel(step=20)

    df_lats = xr.open_dataset(
        "/home/dan/Downloads/icon-eps_global_icosahedral_time-invariant_2024032312_clat.grib2"
    )
    lats = df_lats.tlat

    df_lons = xr.open_dataset(
        "/home/dan/Downloads/icon-eps_global_icosahedral_time-invariant_2024032312_clon.grib2"
    )
    lons = df_lons.tlon

    xy = np.c_[lons, lats]
    z = df.tp.mean(dim="number")

    conversion = utils.swe_to_in("kg m**-2")

    t0 = time.time()
    lut2 = LinearNDInterpolator(xy, z)
    X = np.arange(-180, 180, 0.125)
    Y = np.arange(-90, 90, 0.125)
    X, Y = np.meshgrid(X, Y)
    interped = lut2(X, Y)
    print("interp")
    #plot_precip(X, Y, interped * conversion)
    #print("plot")
    print(time.time() - t0)



def tst():
    member_da = xr.DataArray(data=values[0],
                             coords = {"latitude": lats,
                                       "longitude": lons},
                             dims = ["latitude", "longitude"])
