from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field

class Event(BaseModel):
    """
    Base class for every event on the bus.

    The event_type field is a discriminator: consumers use it to route
    events without isinstance() checks. Pydantic also uses it when
    deserialising mixed event streams from JSON.
    """

    event_type: str = Field(description="Discriminator naming the concrete event type")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Wall-clock time the event was created. Useful for replay logging.",
    )
    seconds_into_session: float = Field(
        description="Simulated time in the race, in seconds from session start"
    )

class LapCompleted(Event):
    """A driver has crossed the line completing a lap."""

    event_type: Literal["LapCompleted"] = "LapCompleted"
    driver_code: str = Field(description="3-letter driver code, e.g. 'LEC'")
    lap_number: int = Field(description="Lap number just completed")
    lap_time_seconds: float | None = Field(
        default=None,
        description="Total lap time in seconds. None if not recorded (formation, dropouts).",
    )
    position: int | None = Field(
        default=None,
        description="Track position at end of this lap. None if not recorded.",
    )

class PitStop(Event):
    """A driver has entered the pit lane to make a tyre change."""

    event_type: Literal["PitStop"] = "PitStop"
    driver_code: str = Field(description="3-letter driver code")
    in_lap: int = Field(description="Lap on which the driver entered the pits")
    out_lap: int | None = Field(
        default=None,
        description="Lap on which the driver rejoined the track. None if not yet rejoined.",
    )
    compound_from: str | None = Field(
        default=None,
        description="Tyre compound coming off, e.g. 'INTERMEDIATE'. None if unknown.",
    )
    compound_to: str | None = Field(
        default=None,
        description="Tyre compound going on. None if not yet known.",
    )

class TrackStatusChange(Event):
    """The track status flag has changed."""

    event_type: Literal["TrackStatusChange"] = "TrackStatusChange"
    lap_number: int = Field(description="Lap on which the status changed")
    new_status: str = Field(
        description="New status: 'green', 'yellow', 'sc', 'vsc', 'red'",
    )

class WeatherUpdate(Event):
    """A new weather observation has been recorded."""

    event_type: Literal["WeatherUpdate"] = "WeatherUpdate"
    air_temp_celsius: float = Field(description="Air temperature in Celsius")
    track_temp_celsius: float = Field(description="Track surface temperature in Celsius")
    humidity_percent: float = Field(description="Relative humidity 0-100")
    is_raining: bool = Field(description="True if rainfall detected")
    wind_speed_ms: float = Field(description="Wind speed in m/s")
