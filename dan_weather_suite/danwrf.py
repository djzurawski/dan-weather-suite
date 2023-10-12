import xarray as xr
from metpy.interpolate import log_interpolate_1d, interpolate_1d
from metpy.units import units
import metpy.calc as mpcalc
from dan_weather_suite.plotting import plot
import matplotlib.pyplot as plt
import cartopy.crs as crs
import numpy as np

from dateutil import parser
from datetime import datetime, timedelta

# GRAVITY = 9.81 * (units.m / units.s**2)


def tst():
    f = "/home/dan/Documents/weather/wrfprd/d01_06"
    f = "/home/dan/Documents/wrf/wrfprd/wrfout_d01_2023-10-12_01:00:00"
    ds = xr.open_dataset(f)
    return ds


def datetime64_to_datetime(dt: np.datetime64) -> datetime:
    return datetime.utcfromtimestamp(int(dt) / 1e9)


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


def vort_500_plot(ds, domain_name):
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

    fhour = int((valid_dt - init_dt).total_seconds() // 3600)

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

    plt.show()


def rh_700_plot(ds, domain_name):
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

    fhour = int((valid_dt - init_dt).total_seconds() // 3600)

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

    plt.show()


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
    fig, ax = plot.add_contourf(
        fig, ax, ds.HGT.XLONG[0], ds.HGT.XLAT[0], ds.HGT[0], levels=levels
    )
    fig.show()


def main():
    f = "/home/dan/Documents/wrf/wrfprd/wrfout_d01_2023-10-12_06:00:00"
    f = "/home/dan/Documents/wrf/wrfprd/wrfout_d02_2023-10-12_06:00:00"
    ds = xr.open_dataset(f)
    ds = preprocess_ds(ds)
    # vort_500_plot(ds, "")
    rh_700_plot(ds, "")


if __name__ == "__main__":
    main()
