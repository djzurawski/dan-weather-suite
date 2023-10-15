import xarray as xr
from metpy.interpolate import log_interpolate_1d, interpolate_1d
from metpy.units import units
import metpy.calc as mpcalc
from dan_weather_suite.plotting import plot
import matplotlib.pyplot as plt
import cartopy.crs as crs
import numpy as np
import argparse
import multiprocessing as mp
from scipy.interpolate import griddata

from dateutil import parser
from datetime import datetime, timedelta
from scipy.spatial import distance

# GRAVITY = 9.81 * (units.m / units.s**2)
CO_NC_DIR = "/home/dan/uems/runs/colorado5km/wrfprd"


def tst():
    # f = "/home/dan/Documents/weather/wrfprd/d01_06"
    # f = "/home/dan/Documents/wrf/wrfprd/wrfout_d01_2023-10-12_01:00:00"
    f = "/home/dan/Documents/wrf/wrfprd/wrfout_d01_2023-10-12_06:00:00"
    ds = xr.open_dataset(f, engine="netcdf4")
    return ds


def datetime64_to_datetime(dt: np.datetime64) -> datetime:
    return datetime.utcfromtimestamp(int(dt) / 1e9)


def domain_netcdf_files(wrf_domain="d02", path=CO_NC_DIR):
    domain_files = sorted([f for f in os.listdir(path) if wrf_domain in f])
    return domain_files


def add_pressure(ds):
    """Adds base pressure and pertubation pressure"""
    pressure_da = ds.PB + ds.P
    pressure_da = pressure_da.assign_attrs(
        {
            "units": "Pa",
            "description": "Actual pressure",
            "MemoryOrder": "XYZ",
            "stagger": "",
        }
    )

    ds["pressure"] = pressure_da
    return ds


def add_geopotential_height(ds):
    """Adds base height plus pertubation height"""
    # raw_units = (units.m**2) / (units.s**2)
    hgt_dm_da = (ds.PH + ds.PHB) / (9.81 * 10)
    hgt_dm_da = hgt_dm_da.assign_attrs(
        {
            "units": "dm",
            "description": "Geopotential height",
            "MemoryOrder": "XYZ",
            "stagger": "Z",
        }
    )
    ds["hgt"] = hgt_dm_da
    return ds


def add_temp_c(ds):
    """Adds temperature in celsius"""

    # 300 adjustment from WRF manual
    theta = (ds.T + 300) * units.degK

    temp_k = mpcalc.temperature_from_potential_temperature(ds.pressure, theta)
    temp_c = temp_k - 273.15
    temp_c = temp_c.assign_attrs(
        {
            "units": "celsius",
            "description": "Temperature",
            "MemoryOrder": "XYZ",
        }
    )
    ds["temp_c"] = temp_c
    return ds


def preprocess_ds(ds):
    ds = add_pressure(ds)
    ds = add_geopotential_height(ds)
    # ds = add_u_mass(ds)
    # ds = add_v_mass(ds)
    ds = add_temp_c(ds)

    return ds


def calc_u_mass(ds):
    um = 0.5 * (ds.U[..., :-1] + ds.U[..., 1:]) * (units.m / units.s)
    return um.values


def calc_v_mass(ds):
    vm = 0.5 * (ds.V[:, :, :-1, :] + ds.V[:, :, 1:, :]) * (units.m / units.s)
    return vm.values


def interpolate_u_stagger_to_mass_coords(da):
    """Converts a z-staggered variable to a mass-staggered variable"""
    mass_da = 0.5 * (da[:, :, :, :-1] + da[:, :, :, 1:])
    return mass_da


def interpolate_v_stagger_to_mass_coords(da):
    """Converts a z-staggered variable to a mass-staggered variable"""
    mass_da = 0.5 * (da[:, :, :-1, :] + da[:, :, 1:, :])
    return mass_da


def interpolate_z_stagger_to_mass_coords(da):
    """Converts a z-staggered variable to a mass-staggered variable"""
    mass_da = 0.5 * (da[:, :-1, :, :] + da[:, 1:, :, :])
    return mass_da


def calc_geopotential_mass(hgt_stag):
    """Calculates geopotential on mass grid"""
    geo_height = 0.5 * (hgt_stag[:, :-1, :, :] + hgt_stag[:, 1:, :, :])
    return geo_height.values


def create_projection(ds):
    """Creates a projection from a wrf dataset"""

    proj = crs.LambertConformal(
        central_latitude=ds.TRUELAT1,
        central_longitude=ds.STAND_LON,
        standard_parallels=(ds.TRUELAT1, ds.TRUELAT2),
    )

    return proj


def downscaled_test(ds, trimmed):
    rain = ds.RAINNC[0]
    grid = interp_to_prism(rain, trimmed)
    wrf_rain = grid.reshape(trimmed.band_data[0].shape)
    wrf_downscaled = wrf_rain * trimmed
    plot_data(
        wrf_downscaled.x, wrf_downscaled.y, np.clip(wrf_downscaled.band_data[0], 0, 20)
    )


def interp_to_prism(ds, ratio):
    xx_ratio, yy_ratio = np.meshgrid(ratio.x, ratio.y)
    ratio_pairs = np.dstack((xx_ratio, yy_ratio)).reshape(-1, 2)
    """
    interp_points = np.dstack((ds.XLONG, ds.XLAT)).reshape(-1,2)
    grid_z0 = griddata(pairs, np.array(ratio.band_data[0]).reshape(-1), interp_points, method='nearest')
    """

    interp_points = ratio_pairs
    wrf_coords = np.dstack((ds.XLONG, ds.XLAT)).reshape(-1, 2)
    grid = griddata(
        wrf_coords, ds.to_numpy().reshape(-1), interp_points, method="cubic"
    )

    data_at_prism_points = grid.reshape(ratio.grid_data.shape[0])

    return data_at_prism_points


def accumulated_precip_plot(ds, domain_name, output_dir, extent=None):
    precip = ds.RAINNC
    lats = ds.PB.XLAT[0]
    lons = ds.PB.XLONG[0]

    projection = create_projection(ds)

    init_dt = parser.parse(ds.START_DATE.replace("_", " "))
    valid_dt = datetime64_to_datetime(ds.XTIME.values[0])
    cycle = str(init_dt.hour).zfill(2)

    fhour = int((valid_dt - init_dt).total_seconds() // 3600)
    fhour_str = str(fhour).zfill(2)

    fig, ax = plot.plot_precip(
        lons.values, lats.values, precip, projection=projection, display_counties=False
    )

    title = plot.make_title_str(
        init_dt,
        valid_dt,
        fhour,
        "Acc precip",
        "danwrf",
        "(in)",
    )
    ax.set_title(title)

    fname = f"{output_dir}/danwrf.{cycle}z.{domain_name}.vort500.f{fhour_str}.png"
    fig.savefig(fname, bbox_inches="tight")


def vort_500_plot(ds, domain_name, output_dir):
    levels = [500] * units.hPa

    ds = add_pressure(ds)
    ds = add_geopotential_height(ds)

    u_sigma = calc_u_mass(ds) * units.m / units.s
    v_sigma = calc_v_mass(ds) * units.m / units.s
    press = ds.pressure

    hgt_mass = calc_geopotential_mass(ds.hgt)

    height_levels, u_levels, v_levels = log_interpolate_1d(
        levels, press, hgt_mass, u_sigma, v_sigma, axis=1
    )

    u_kt = u_levels.to("kt")
    v_kt = v_levels.to("kt")

    height_levels_dm = height_levels

    lats = ds.PB.XLAT[0]
    lons = ds.PB.XLONG[0]

    dx = ds.DX * units.m
    dy = ds.DY * units.m
    vort = mpcalc.vorticity(u_levels, v_levels, dx=dx, dy=dy)

    vort = vort * 1e5

    projection = create_projection(ds)

    init_dt = parser.parse(ds.START_DATE.replace("_", " "))
    valid_dt = datetime64_to_datetime(ds.XTIME.values[0])
    cycle = str(init_dt.hour).zfill(2)

    fhour = int((valid_dt - init_dt).total_seconds() // 3600)
    fhour_str = str(fhour).zfill(2)

    fig, ax = plot.plot_500_vorticity(
        lons.values,
        lats.values,
        height_levels_dm[0][0],
        vort[0][0],
        u_kt[0][0],
        v_kt[0][0],
        projection=projection,
        display_counties=False,
    )

    title = plot.make_title_str(
        init_dt,
        valid_dt,
        fhour,
        "500mb vorticity",
        "danwrf",
        " (10^5 s^-1)",
    )
    ax.set_title(title)

    fname = f"{output_dir}/danwrf.{cycle}z.{domain_name}.vort500.f{fhour_str}.png"
    fig.savefig(fname, bbox_inches="tight")


def rh_700_plot(ds, domain_name, output_dir):
    levels = [700] * units.hPa

    ds = add_pressure(ds)
    ds = add_geopotential_height(ds)

    u_sigma = calc_u_mass(ds) * units.m / units.s
    v_sigma = calc_v_mass(ds) * units.m / units.s
    t_sigma = ds.temp_c
    q_sigma = ds.QVAPOR
    press = ds.pressure

    hgt_mass = calc_geopotential_mass(ds.hgt)

    height_levels, u_levels, v_levels, t_levels, q_levels = log_interpolate_1d(
        levels, press, hgt_mass, u_sigma, v_sigma, t_sigma, q_sigma, axis=1
    )

    u_kt = u_levels.to("kt")
    v_kt = v_levels.to("kt")

    height_levels_dm = height_levels

    lats = ds.PB.XLAT[0]
    lons = ds.PB.XLONG[0]

    rh_700 = mpcalc.relative_humidity_from_specific_humidity(
        700 * units.hPa, t_levels, q_levels
    )
    rh_700 = rh_700 * 100
    rh_700 = np.clip(rh_700, 0, 100)

    projection = create_projection(ds)

    init_dt = parser.parse(ds.START_DATE.replace("_", " "))
    valid_dt = datetime64_to_datetime(ds.XTIME.values[0])
    cycle = str(init_dt.hour).zfill(2)

    fhour = int((valid_dt - init_dt).total_seconds() // 3600)
    fhour_str = str(fhour).zfill(2)

    fig, ax = plot.plot_700_rh(
        lons.values,
        lats.values,
        height_levels_dm[0][0],
        rh_700[0][0],
        u_kt[0][0],
        v_kt[0][0],
        projection=projection,
        display_counties=False,
    )

    title = plot.make_title_str(
        init_dt,
        valid_dt,
        fhour,
        "700mg relative humidity",
        "danwrf",
        "%",
    )
    ax.set_title(title)
    fname = f"{output_dir}/danwrf.{cycle}z.{domain_name}.rh700.f{fhour_str}.png"
    fig.savefig(fname, bbox_inches="tight")


def terp():
    f = "/home/dan/Documents/weather/wrfprd/d01_06"
    ds = xr.open_dataset(f)
    ds = add_geopotential_height(ds)
    ds = add_pressure(ds)
    hgt_mass = calc_geopotential_mass(ds.hgt)

    p = ds.pressure[0, :, 20, 20]
    h = hgt_mass[0, :, 20, 20]

    levels = np.arange(10, 10000, 10) * 100 * units.Pa
    levels = np.array([1000, 850, 700, 500, 200, 100]) * 100 * units.Pa

    height = interpolate_1d(levels, ds.pressure, hgt_mass, axis=1)

    plt.plot(p, h, marker=".")
    plt.plot(levels, height[0, :, 20, 20], marker="x", label="linear")

    height = log_interpolate_1d(levels, ds.pressure, hgt_mass, axis=1)
    plt.plot(levels, height[0, :, 20, 20], marker="o", label="llog")
    plt.legend()

    # plt.plot(ds.hgt[0,:,20,20], marker='.')
    # plt.plot(hgt_mass[0,:,20,20], marker='.')
    plt.show()


def terrain(ds):
    projection = create_projection(ds)
    fig, ax = plot.create_basemap(display_counties=True, projection=projection)
    levels = np.arange(1000, 3800, 100)

    ax.pcolormesh(
        ds.HGT.XLONG[0],
        ds.HGT.XLAT[0],
        ds.HGT[0],
        vmin=1000,
        vmax=3800,
        transform=crs.PlateCarree(),
    )

    plt.show()


def vort_500_plots(wrfprd_dir, domain_name, wrf_domain="d01"):
    nc_paths = [
        wrfprd_dir + "/" + nc_file
        for nc_file in domain_netcdf_files(path=wrfprd_dir, wrf_domain=wrf_domain)
    ]

    for nc_path in nc_paths:
        vort_500_plot(nc_path, domain_name)


def rh_700_plots(wrfprd_dir, domain_name, wrf_domain):
    nc_paths = [
        wrfprd_dir + "/" + nc_file
        for nc_file in domain_netcdf_files(path=wrfprd_dir, wrf_domain=wrf_domain)
    ]

    for nc_path in nc_paths:
        rh_700_plot(nc_path, domain_name)


def main(wrfprd_path, domain_names, wrf_domains=["d01"], labels=[]):
    # Do it this way because mp.Pool() freezes computer when using after calling
    # accumulated_swe_plots()
    with mp.Pool() as pool:
        for domain_name, wrf_domain in zip(domain_names, wrf_domains):
            pool.apply_async(
                accumulated_swe_plots,
                (wrfprd_path, domain_name, wrf_domain, labels),
                error_callback=error_callback,
            )
            pool.apply_async(
                accumulated_precip_plots,
                (wrfprd_path, domain_name, wrf_domain, labels),
                error_callback=error_callback,
            )
            pool.apply_async(
                rh_700_plots,
                (wrfprd_path, domain_name, wrf_domain),
                error_callback=error_callback,
            )
            pool.apply_async(
                vort_500_plots,
                (wrfprd_path, domain_name, wrf_domain),
                error_callback=error_callback,
            )

            pool.apply_async(
                temp_2m_plots,
                (wrfprd_path, domain_name, wrf_domain),
                error_callback=error_callback,
            )

        pool.close()
        pool.join()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Danwrf plot generator")
    argparser.add_argument("-p", "--wrfprd-path", type=str, required=True)
    argparser.add_argument("-d", "--domain-name", type=str, required=True)
    argparser.add_argument("-n", "--num-nests", type=int, default=1)

    args = argparser.parse_args()

    domain_name = args.domain_name
    wrfprd_path = args.wrfprd_path
    num_nests = args.num_nests

    wrf_domains = ["d0" + str(i) for i in range(1, num_nests + 1)]
    domain_names = [f"{domain_name}-{d}" for d in wrf_domains]
