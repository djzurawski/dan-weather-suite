import climetlab as cml
from datetime import datetime, timedelta
import numpy as np


def set_init_dt(utc_dt: datetime):
    """Empirically figured out the most recent available GFS cycle"""
    utc_hour = utc_dt.hour
    if utc_hour >= 4 and utc_hour < 10:
        utc_dt = utc_dt.replace(hour=0)
    elif utc_hour >= 10 and utc_hour < 16:
        utc_dt = utc_dt.replace(hour=6)
    elif utc_hour >= 16 and utc_hour < 20:
        utc_dt = utc_dt.replace(hour=12)
    else:
        utc_dt = utc_dt.replace(hour=18)
        utc_dt = utc_dt - timedelta(days=1)

    return utc_dt


def load_gfs(init_dt: datetime):
    year = str(init_dt.year)
    month = str(init_dt.month).zfill(2)
    day = str(init_dt.day).zfill(2)
    hour = str(init_dt.hour).zfill(2)

    date = year + month + day

    data = cml.load_source(
        "url-pattern",
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/"
        "gfs.{date}/{hour}/atmos/gfs.t{hour}z.pgrb2.0p25.f000",
        date=date,
        hour=hour,
    )

    return data


def create_pangu_grib(gfs_data):
    params_surface = ["prmsl", "10u", "10v", "2t"]
    params_pressure_levels = ["gh", "q", "t", "u", "v"]
    pressure_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

    fields_pl = gfs_data.sel(
        param=params_pressure_levels,
        level=pressure_levels,
        levtype="pl",
    )

    fields_sfc = gfs_data.sel(
        param=params_surface,
        levtype="sfc",
    )

    fields = fields_pl + fields_sfc

    RENAME_DICT = {"gh": "z", "prmsl": "msl"}
    UNIT_CONVERSION_DICT = {"gh": 9.80665}

    out = cml.new_grib_output("gfsinput.grib")

    for f in fields:
        param = f.metadata("shortName")
        out.write(
            f.to_numpy() * UNIT_CONVERSION_DICT.get(param, 1),
            template=f,
            centre=98,
            setLocalDefinition=1,
            subCentre=7,
            localDefinitionNumber=1,
            param=RENAME_DICT.get(param, param),
        )


if __name__ == "__main__":
    utc_dt = datetime.utcnow()
    utc_dt = set_init_dt(utc_dt)
    gfs_data = load_gfs(utc_dt)
    create_pangu_grib(gfs_data)
