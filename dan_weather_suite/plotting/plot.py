import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature
from cartopy.feature import STATES

import matplotlib.colors as mcolors
import numpy as np

M_PER_S_TO_KT = 1.94384
MM_TO_IN = 0.03937008

VORT_COLORS = (
    np.array(
        [
            (255, 255, 255),
            (190, 190, 190),
            (151, 151, 151),
            (131, 131, 131),
            (100, 100, 100),
            (0, 255, 255),
            (0, 231, 205),
            (0, 203, 126),
            (0, 179, 0),
            (126, 205, 0),
            (205, 231, 0),
            (255, 255, 0),
            (255, 205, 0),
            (255, 153, 0),
            (255, 102, 0),
            (255, 0, 0),
            (205, 0, 0),
            (161, 0, 0),
            (141, 0, 0),
            (121, 0, 0),
            (124, 0, 102),
            (145, 0, 155),
            (163, 0, 189),
            (255, 0, 231),
            (255, 201, 241),
        ]
    )
    / 255.0
)

VORT_LEVELS = [
    0.5,
    1,
    1.5,
    2,
    3,
    4,
    5,
    6,
    8,
    10,
    12,
    14,
    16,
    18,
    20,
    22,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
    85,
]

PRECIP_CLEVS = [
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1,
    1.2,
    1.4,
    1.6,
    1.8,
    2.0,
    2.5,
    3,
    3.5,
    4,
    5,
    6,
    7,
    8,
    9,
]

PRECIP_CMAP_DATA = (
    np.array(
        [
            (190, 190, 190),
            (170, 165, 165),
            (130, 130, 130),
            (110, 110, 110),
            (180, 250, 170),
            (150, 245, 140),
            (120, 245, 115),
            (80, 240, 80),
            (30, 180, 30),
            (15, 160, 15),
            (20, 100, 210),
            (40, 130, 240),
            (80, 165, 245),
            (150, 210, 230),
            (180, 240, 250),
            (255, 250, 170),
            (255, 232, 120),
            (255, 192, 60),
            (253, 159, 0),
            (255, 96, 0),
            (253, 49, 0),
            (225, 20, 20),
            (191, 0, 0),
            (165, 0, 0),
            (135, 0, 0),
            (99, 59, 59),
            (139, 99, 89),
            (179, 139, 129),
            (199, 159, 149),
            (240, 240, 210),
        ]
    )
    / 255.0
)

RH_LEVELS = [
    0,
    1,
    2,
    3,
    5,
    10,
    15,
    20,
    25,
    30,
    40,
    50,
    60,
    65,
    70,
    75,
    80,
    85,
    90,
    95,
    99,
    100,
]


def get_barb_interval(domain):
    if domain == "d02":
        return 6
    else:
        return 8


def coriolis_parameter(lat_degrees):
    lat_rads = lat_degrees * (np.pi / 180)
    f = 2 * 7.2921e-5 * np.sin(lat_rads)
    return f


def make_title_str(init_dt, valid_dt, fhour, field_name, model_name="", field_units=""):
    date_format = "%Y-%m-%dT%HZ"
    init_str = init_dt.strftime(date_format)
    valid_str = valid_dt.strftime(date_format)
    fhour = str(fhour).zfill(2)

    return f"{model_name}   Init: {init_str}    Valid: {valid_str}    {field_name} ({field_units})   Hour: {fhour}"


def add_title(
    fig, ax, init_dt, valid_dt, fhour, field_name, model_name="", field_units=""
):
    text = make_title_str(init_dt, valid_dt, fhour, field_name, model_name, field_units)

    fig.title(text)
    return fig, ax


def create_basemap(projection=crs.PlateCarree()):
    fig, ax = plt.subplots(figsize=(18, 10), subplot_kw={"projection": projection})
    # fig = plt.figure(figsize=(18, 10))
    # ax = plt.axes(projection=projection)

    border_scale = "50m"
    county_scale = "20m"

    ax.coastlines(border_scale, linewidth=0.8)

    lakes = NaturalEarthFeature(
        "physical", "lakes", border_scale, edgecolor="blue", facecolor="none"
    )
    ax.add_feature(lakes, facecolor="none", edgecolor="blue", linewidth=0.5)
    ax.add_feature(STATES, edgecolor="black")

    # ax.add_feature(USCOUNTIES.with_scale(county_scale), edgecolor="gray")
    # ax.add_feature(USSTATES.with_scale(county_scale), edgecolor="black")

    return fig, ax


def add_contour(
    fig,
    ax,
    lons,
    lats,
    data,
    levels=None,
):
    contours = ax.contour(
        lons,
        lats,
        data,
        levels=levels,
        colors="black",
        transform=crs.PlateCarree(),
    )
    ax.clabel(contours, inline=1, fontsize=10, fmt="%i")
    return fig, ax


def add_contourf(
    fig,
    ax,
    lons,
    lats,
    data,
    levels=None,
    colors=None,
    cmap=None,
):
    if colors is not None and levels is not None:
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(levels, cmap.N)
    elif cmap is not None and levels is not None:
        norm = mcolors.BoundaryNorm(levels, cmap.N)
    else:
        cmap = None
        norm = None

    contours = ax.contourf(
        lons,
        lats,
        data,
        levels=levels,
        norm=norm,
        cmap=cmap,
        transform=crs.PlateCarree(),
    )

    """
    contours = ax.pcolormesh(
        lons,
        lats,
        data,
        # levels,
        # levels=levels,
        norm=norm,
        cmap=cmap,
        transform=crs.PlateCarree(),
    )
    """
    fig.colorbar(contours, ax=ax, orientation="vertical", pad=0.05)
    return fig, ax


def add_wind_barbs(
    fig,
    ax,
    lons,
    lats,
    u,
    v,
    barb_length=5.5,
    # barb_interval=8,
):
    # step = barb_interval
    step = int(lons.shape[0] // 20)

    u = np.array(u)
    v = np.array(v)

    ax.barbs(
        lons[::step, ::step],
        lats[::step, ::step],
        u[::step, ::step],
        v[::step, ::step],
        transform=crs.PlateCarree(),
        length=barb_length,
    )
    return fig, ax


def add_label_markers(fig, ax, labels):
    """labels: ('text', (lon, lat))"""
    for label in labels:
        text, coords = label
        lon, lat = coords
        ax.text(lon, lat, text, horizontalalignment="left", transform=crs.PlateCarree())
        ax.plot(
            lon, lat, markersize=2, marker="o", color="k", transform=crs.PlateCarree()
        )

    return fig, ax


def plot_precip(lons, lats, precip_in, **kwargs):
    projection = kwargs.get("projection", crs.PlateCarree())

    fig, ax = create_basemap(projection=projection)

    fig, ax = add_contourf(
        fig,
        ax,
        lons,
        lats,
        precip_in,
        PRECIP_CLEVS,
        PRECIP_CMAP_DATA,
    )

    if "u10" in kwargs and "v10" in kwargs:
        u10 = kwargs["u10"]
        v10 = kwargs["v10"]
        fig, ax = add_wind_barbs(fig, ax, lons, lats, u10, v10)

    return fig


def plot_swe(lons, lats, swe_in, **kwargs):
    projection = kwargs.get("projection", crs.PlateCarree())

    fig, ax = create_basemap(projection=projection)

    fig, ax = add_contourf(
        fig,
        ax,
        lons,
        lats,
        swe_in,
        PRECIP_CLEVS,
        PRECIP_CMAP_DATA,
    )

    if "u10" in kwargs and "v10" in kwargs:
        u10 = kwargs["u10"]
        v10 = kwargs["v10"]
        fig, ax = add_wind_barbs(fig, ax, lons, lats, u10, v10)

    if "labels" in kwargs:
        labels = kwargs["labels"]
        fig, ax = add_label_markers(fig, ax, labels)

    return fig, ax


def plot_temp_2m(lons, lats, temp, **kwargs):
    projection = kwargs.get("projection", crs.PlateCarree())

    fig, ax = create_basemap(projection=projection)

    # cmap=get_cmap('gist_ncar')
    # cmap = get_cmap("nipy_spectral")
    cmap = get_cmap("jet")
    levels = np.arange(-30, 110, 5)

    fig, ax = add_contourf(fig, ax, lons, lats, temp, levels=levels, cmap=cmap)

    if "u10" in kwargs and "v10" in kwargs:
        u10 = kwargs["u10"]
        v10 = kwargs["v10"]
        fig, ax = add_wind_barbs(fig, ax, lons, lats, u10, v10)

    if "labels" in kwargs:
        labels = kwargs["labels"]
        fig, ax = add_label_markers(fig, ax, labels)

    return fig, ax


def plot_500_vorticity(lons, lats, hgt_500, vort_500, u_500, v_500, **kwargs):
    projection = kwargs.get("projection", crs.PlateCarree())

    fig, ax = create_basemap(projection=projection)

    hgt_500_levels = np.arange(492, 594, 3)

    fig, ax = add_contour(fig, ax, lons, lats, hgt_500, hgt_500_levels)

    fig, ax = add_contourf(
        fig,
        ax,
        lons,
        lats,
        vort_500,
        levels=VORT_LEVELS,
        colors=VORT_COLORS,
    )

    fig, ax = add_wind_barbs(fig, ax, lons, lats, u_500, v_500)

    if "title" in kwargs:
        ax.set_title(kwargs["title"])

    return fig, ax


def plot_700_rh(lons, lats, hgt_700, rh_700, u_700, v_700, **kwargs):
    projection = kwargs.get("projection", crs.PlateCarree())
    fig, ax = create_basemap(projection=projection)

    hgt_700_levels = np.arange(180, 420, 3)

    rh_clevels = [
        0,
        1,
        2,
        3,
        5,
        10,
        15,
        20,
        25,
        30,
        40,
        50,
        60,
        65,
        70,
        75,
        80,
        85,
        90,
        95,
        99,
        100,
    ]

    fig, ax = add_contour(
        fig,
        ax,
        lons,
        lats,
        hgt_700,
        hgt_700_levels,
    )

    fig, ax = add_contourf(
        fig,
        ax,
        lons,
        lats,
        rh_700,
        levels=rh_clevels,
        cmap=get_cmap("BrBG"),
    )

    fig, ax = add_wind_barbs(
        fig,
        ax,
        lons,
        lats,
        u_700,
        v_700,
    )

    if "title" in kwargs:
        ax.set_title(kwargs["title"])

    return fig, ax
