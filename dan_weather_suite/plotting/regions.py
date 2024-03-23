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


# New Mexico
SANDIA_PEAK = Label(text="Sandia Peak", lat=35.2, lon=-106.429)
SKI_SANTA_FE = Label(text="Ski Santa Fe", lat=35.79, lon=-105.789)
TAOS = Label(text="Taos Ski Valley", lat=36.573, lon=-105.448)

# Colorado
ABASIN = Label(text="Abasin", lat=39.635, lon=-105.871)
ASPEN = Label(text="Aspen-Snowmass", lat=39.175, lon=-106.959)
BERTHOUD_PASS = Label(text="Berthoud Pass", lat=39.781, lon=-105.777)
BOULDER = Label(text="Boulder", lat=40.01, lon=-105.27)
BRECKENRIDGE = Label(text="Breckenridge", lat=39.478, lon=-106.077)
COPPER = Label(text="Copper", lat=39.485, lon=-106.16)
CRESTED_BUTTE = Label(text="Crested Butte", lat=38.888, lon=-106.943)
ELDORA = Label(text="Eldora", lat=39.94, lon=-105.596)
KEYSTONE = Label(text="Keystone", lat=39.58, lon=-105.94)
LOVELAND = Label(text="Loveland", lat=39.673, lon=-105.908)
PURGATORY = Label(text="Purgatory", lat=37.623, lon=-107.838)
STEAMBOAT = Label(text="Steamboat", lat=40.46, lon=-106.76)
SILVERTON = Label(text="Silverton", lat=37.877, lon=-107.653)
TELLURIDE = Label(text="Telluride", lat=37.911, lon=-107.822)
WINTER_PARK = Label(text="Winter Park", lat=39.858, lon=-105.773)
WOLF_CREEK = Label(text="Wolf Creek", lat=37.47, lon=-106.80)
VAIL = Label(text="Vail", lat=39.61, lon=-106.375)
TOWER_SNOTEL = Label(text="Tower SNOTEL, CO", lat=40.54, lon=-106.68)

# Utah
ALTA_SNOWBIRD = Label(text="Alta-Snowbird", lat=40.577, lon=-111.63)
BRIGHTON = Label(text="Brighton", lat=40.60, lon=-111.57)
PARK_CITY = Label(text="Park City-Deer Valley", lat=40.625, lon=-111.5)

PARK_CITY_SUMMIT = Label(text="Park City Summit", lat=40.61, lon=-111.55)
PARK_CITY_BASE = Label(text="Park City Base", lat=40.65, lon=-111.52)
DEER_VALLEY = Label(text="Deer Valley", lat=40.62, lon=-111.50)

POWDER_MOUNTAIN = Label(text="Powder Mtn", lat=41.38, lon=-111.78)
SNOWBASIN = Label(text="Snowbasin", lat=41.2, lon=-111.863)

# Montana
BIG_SKY = Label(text="Big Sky", lat=45.27, lon=-111.45)

MAMMOTH = Label(text="Mammoth", lat=37.641, lon=-119.024)

# Wyoming
GRAND_TARGHEE = Label(text="Grand Targhee", lat=43.787, lon=-110.936)
JACKSON_HOLE_R = Label(text="Jackson Hole Rendezvous Bowl", lat=43.59, lon=-110.87)
JACKSON_HOLE = Label(text="Jackson Hole Mid Mtn", lat=43.598, lon=-110.840)


FRONT_RANGE_EXTENT = Extent(top=41.4, bottom=37.75, left=-108.4, right=-104.1)

FRONT_RANGE_LABELS = [ABASIN, BOULDER, COPPER, ELDORA, STEAMBOAT, VAIL, WINTER_PARK]


FRONT_RANGE = Region(
    name="frange",
    extent=FRONT_RANGE_EXTENT,
    labels=FRONT_RANGE_LABELS,
    display_counties=True,
)

COLORADO_EXTENT = Extent(top=42, bottom=36, left=-109.5, right=-101.5)
WASATCH_EXTENT = Extent(top=42.0, bottom=39.5, left=-113.0, right=-110.5)

WASATCH_LABELS = [POWDER_MOUNTAIN, SNOWBASIN, ALTA_SNOWBIRD, PARK_CITY]


COLORADO = Region(name="colorado", extent=COLORADO_EXTENT, display_counties=True)
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


DANSEMBLE_SELECT = sorted(
    [
        PURGATORY,
        TELLURIDE,
        SILVERTON,
        MAMMOTH,
        ASPEN,
        CRESTED_BUTTE,
        SANDIA_PEAK,
        SKI_SANTA_FE,
        TAOS,
        ABASIN,
        BERTHOUD_PASS,
        BOULDER,
        BRECKENRIDGE,
        COPPER,
        ELDORA,
        KEYSTONE,
        LOVELAND,
        STEAMBOAT,
        WINTER_PARK,
        WOLF_CREEK,
        VAIL,
        TOWER_SNOTEL,
        ALTA_SNOWBIRD,
        BRIGHTON,
        PARK_CITY_SUMMIT,
        PARK_CITY_BASE,
        DEER_VALLEY,
        POWDER_MOUNTAIN,
        SNOWBASIN,
        BIG_SKY,
        GRAND_TARGHEE,
        JACKSON_HOLE,
        JACKSON_HOLE_R,
    ],
    key=lambda x: x.text,
)
