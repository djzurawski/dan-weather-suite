from pydantic import BaseModel, Field, model_validator, computed_field
from typing import Sequence


LatitudeField = Field(ge=-90, le=90)
LongitudeField = Field(ge=-180, le=180)


class Label(BaseModel):
    text: str
    lat: float = LatitudeField
    lon: float = LongitudeField


class Extent(BaseModel):
    top: float = LatitudeField
    bottom: float = LatitudeField
    left: float = LongitudeField
    right: float = LongitudeField

    @model_validator(mode="after")
    def check_extent_valid(self):
        "Ensures top > bottom, right > left"
        if self.bottom >= self.top:
            raise ValueError("top must be greater than bottom")
        if self.left >= self.right:
            raise ValueError("right must be greater than left")
        return self

    @computed_field
    def central_longitude(self) -> float:
        return (self.left + self.right) / 2


class Region(BaseModel):
    name: str
    extent: Extent
    labels: Sequence[Label] = []
    display_counties: bool = False


FRONT_RANGE_EXTENT = Extent(top=40.9, bottom=38.25, left=-107.9, right=-104.6)

FRONT_RANGE_LABELS = [
    Label(text="Abasin", lat=39.635, lon=-105.871),
    Label(text="Boulder", lat=40.01, lon=-105.27),
    Label(text="Copper", lat=39.485, lon=-106.16),
    Label(text="Eldora", lat=39.94, lon=-105.595),
    Label(text="Steamboat", lat=40.46, lon=-106.76),
    Label(text="Vail", lat=39.61, lon=-106.375),
    Label(text="Winter Park", lat=39.867, lon=-105.77),
]


FRONT_RANGE = Region(
    name="frange",
    extent=FRONT_RANGE_EXTENT,
    labels=FRONT_RANGE_LABELS,
    display_counties=True,
)


WASATCH_EXTENT = Extent(top=41.5, bottom=40, left=-112.5, right=-111.0)

WASATCH_LABELS = [
    Label(text="Powder Mtn", lat=41.38, lon=-111.78),
    Label(text="Snowbasin", lat=41.2, lon=-111.855),
    Label(text="Alta-Snowbird", lat=40.577, lon=-111.63),
    Label(text="Park City-Deer Valley", lat=40.625, lon=-111.5),
]


WASATCH = Region(
    name="wasatch", extent=WASATCH_EXTENT, labels=WASATCH_LABELS, display_counties=True
)

HREF_REGIONS = [FRONT_RANGE, WASATCH]


CONUS_EXTENT = Extent(top=61, bottom=15, left=-130, right=-65)

WEST_CONUS_EXTENT = Extent(left=-130, right=-95, bottom=25, top=55)

PANGU_NA_EXTENT = Extent(top=72, bottom=15, left=-180, right=-40)


PRISM_EXTENT = Extent(
    top=49.93333367, bottom=24.06666701, left=-125.01666667, right=-66.48333336
)

SKI_TAOS = Label(text="Taos", lat=36.573, lon=-105.448)
SKI_SANTA_FE = Label(text="Ski Santa Fe", lat=35.79, lon=-105.789)
SANDIA_PEAK = Label(text="Sandia Peak", lat=35.2, lon=-106.429)

BRECK = Label(text="Breck", lat=39.478, lon=-106.077)
WINTER_PARK = Label(text="Winter Park", lat=39.867, lon=-105.77)
BERTHOUD = Label(text="Winter Park", lat=39.80, lon=-105.77)
