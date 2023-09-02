from pydantic import BaseModel, Field, model_validator, computed_field

from typing import Iterable


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

    @model_validator(mode="before")
    @classmethod
    def check_event_valid(cls, data):
        "Ensures top > bottom, right > left"
        if isinstance(data, dict):
            top = data.get("top")
            bottom = data.get("bottom")
            left = data.get("left")
            right = data.get("right")
            if bottom >= top:
                raise ValueError("top must be greater than bottom")
            if left >= right:
                raise ValueError("right must be greater than left")
        return data

    @computed_field
    def central_longitude(self) -> float:
        return (self.left + self.right) / 2


class Region(BaseModel):
    name: str
    extent: Extent
    labels: Iterable[Label] = []
    display_counties: bool = False


FRONT_RANGE_EXTENT = Extent(top=40.9, bottom=38.25, left=-107.9, right=-104.6)

FRONT_RANGE_LABELS = [
    Label(text="Boulder", lat=40.01, lon=-105.27),
    Label(text="Winter Park", lat=39.867, lon=-105.77),
]

FRONT_RANGE = Region(
    name="Front Range",
    extent=FRONT_RANGE_EXTENT,
    labels=FRONT_RANGE_LABELS,
    display_counties=True,
)
