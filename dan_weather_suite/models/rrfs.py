import boto3
from botocore import UNSIGNED
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor
from dan_weather_suite.models.loader import ModelLoader
from datetime import datetime, time, timedelta
import numpy as np
import os
import pandas as pd
import pickle

from scipy.spatial import KDTree
import xarray as xr

S3_BUCKET = "noaa-rrfs-pds"


def format_member_prefix(init_dt: datetime, fhour: int, member: int) -> str:
    "member 0 is control"

    day_str = init_dt.strftime("%Y%m%d")
    cycle_str = str(init_dt.hour).zfill(2)
    fhour_str = str(fhour).zfill(3)
    if member == 0:
        prefix = (
            f"rrfs_a/rrfs_a.{day_str}/{cycle_str}/control/rrfs.t{cycle_str}z"
            f".prslev.f{fhour_str}.conus_3km.grib2"
        )

    else:
        prefix = (
            f"rrfs_a/rrfs_a.{day_str}/{cycle_str}/mem000{member}/rrfs"
            f".t{cycle_str}z.prslev.f{fhour_str}.conus_3km.grib2"
        )

    return prefix


def parse_grib_idx(s3_path: str) -> pd.DataFrame:
    column_names = [
        "message",
        "start_byte",
        "init",
        "variable",
        "level",
        "forecast_time",
        "unk1",
        "unk2",
        "unk3",
    ]
    df = pd.read_csv(s3_path, sep=":", names=column_names)
    df["end_byte"] = df.shift(-1)["start_byte"] - 1

    return df


def extract_acc_precip_byte_range(df: pd.DataFrame, fhour: int) -> str:
    variable = "APCP"

    if fhour % 24 == 0:
        days = fhour // 24
        forecast_time = f"0-{days} day acc fcst"
    else:
        forecast_time = f"0-{fhour} hour acc fcst"

    precip_df = df[df["variable"] == variable]
    acc_precip = precip_df[precip_df["forecast_time"] == forecast_time]

    start_byte = int(acc_precip.iloc[0].start_byte)
    end_byte = int(acc_precip.iloc[0].end_byte)

    return f"{start_byte}-{end_byte}"


def download_member_grib(init_dt: datetime, member: int, flength: int = 48) -> str:
    grib_file = f"grib/rrfs-mem{member}.grib"
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    if os.path.exists(grib_file):
        os.remove(grib_file)

    with open(grib_file, "ab") as f:
        for fhour in range(1, flength + 1):
            prefix = format_member_prefix(init_dt, fhour, member)
            idx_s3_path = f"s3://{S3_BUCKET}/{prefix}.idx"
            idx_df = parse_grib_idx(idx_s3_path)
            acc_precip_range = extract_acc_precip_byte_range(idx_df, fhour)
            print(prefix, acc_precip_range)
            byte_range = f"bytes={acc_precip_range}"
            response = s3.get_object(Bucket=S3_BUCKET, Key=prefix, Range=byte_range)
            data = response["Body"].read()
            f.write(data)

    return grib_file


class RrfsLoader(ModelLoader):
    def __init__(self):
        super().__init__()
        self.step_size = 1
        self.forecast_length = 60
        self.forecast_hours = list(
            range(1, self.forecast_length + self.step_size, self.step_size)
        )
        self.grib_file = "grib/rrfs.grib"
        self.netcdf_file = "grib/rrfs.nc"
        self.kd_tree = "rrfs-tree.pkl"
        self.bucket = S3_BUCKET
        self.num_members = 6

    def get_latest_init(self) -> datetime:
        current_utc = datetime.utcnow()
        current_utc_time = current_utc.time()

        release_00z = time(7, 00)
        release_12z = time(19, 00)

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

        with ThreadPoolExecutor() as pool:
            futures = []
            members = range(self.num_members)
            for member in members:
                future = pool.submit(
                    download_member_grib, init_dt, member, self.forecast_length
                )
                futures.append(future)

            for i, f in enumerate(futures):
                try:
                    print(f.result())
                except Exception as e:
                    print(f"RRFS download error {e}")
                    grib_file = f"grib/rrfs-mem{i}.grib"
                    if os.path.exists(grib_file):
                        os.remove(grib_file)

    def load_member_grib(self, member: int) -> xr.Dataset:
        grib_file = f"grib/rrfs-mem{member}.grib"
        return xr.open_dataset(grib_file, chunks={})

    def combine_members(self) -> xr.Dataset:
        members = range(self.num_members)
        datasets = []
        for i in members:
            try:
                datasets.append(self.load_member_grib(i))
            except Exception as e:
                print(f"Count not combine member {i}: {e}")

        for i, ds in enumerate(datasets):
            datasets[i] = ds.expand_dims({"number": [i]})

        combined_ds = xr.concat(datasets, dim="number")
        return combined_ds

    def process_grib(self) -> xr.Dataset:
        ds = self.combine_members()
        ds["longitude"] = (ds["longitude"] + 180) % 360 - 180
        return ds

    def create_kdtree(self) -> KDTree:
        "Create KD tree of ds coordinates"
        ds = self.open_dataset()
        pairs = np.dstack((ds.longitude, ds.latitude)).reshape(-1, 2)
        tree = KDTree(pairs)
        with open(self.kd_tree, "wb") as f:
            pickle.dump(tree, f)

        return tree

    def get_kdtree(self) -> KDTree:
        "Create KD tree of ds coordinates"
        if not os.path.exists(self.kd_tree):
            tree = self.create_kdtree()
        else:
            with open(self.kd_tree, "rb") as f:
                tree = pickle.load(f)
        return tree
