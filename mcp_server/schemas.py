"""
Pydantic models defining the return shapes of MCP tools.

These models are how the LLM understands what each tool returns.
Field names, types, and docstrings all flow into the JSON schema
that the LLM reads, so they need to be clear and precise.
"""
from pydantic import BaseModel, Field

class SessionState(BaseModel):
    """High-level snapshot of a session at a given lap."""

    year: int = Field(description="Championship year, e.g. 2022")
    event: str = Field(description="Event name, e.g. 'Monaco Grand Prix'")
    session_type: str = Field(description="Session type code: R, Q, FP1, FP2, FP3, S")
    total_laps: int = Field(description="Total race laps for this event")
    current_lap: int = Field(description="The lap number we are currently simulating up to")
    leader_driver_code: str = Field(description="3-letter code of the driver currently leading at current_lap, e.g. 'VER'")
    leader_lap_time_seconds: float | None = Field(description="Leader's lap time at current_lap, in seconds. None if not available.")
    track_status: str = Field(description="Track status flag: 'green', 'yellow', 'sc' (safety car), 'vsc' (virtual SC), 'red'")

class LapRecord(BaseModel):
    """Record of one driver for one lap. Used for stint analysis"""

    lap_number: int = Field(description="Lap number, 1-indexed")
    lap_time_seconds: float = Field(description="Total lap time in seconds")
    sector_1_seconds: float | None = Field(description="Sector 1 time in seconds, or None if not recorded")
    sector_2_seconds: float | None = Field(description="Sector 2 time in seconds, or None if not recorded")
    sector_3_seconds: float | None = Field(description="Sector 3 time in seconds, or None if not recorded")
    compound: str = Field(description="Tyre compound on this lap: SOFT, MEDIUM, HARD, INTERMEDIATE, WET")
    tyre_life: int = Field(description="Number of laps this set of tyres has done by the end of this lap")
    stint: int = Field(description="Stint number, 1-indexed. Increments after each pit stop.")
    is_pit_in_lap: bool = Field(description="True if the driver entered the pit lane at the end of this lap")
    is_pit_out_lap: bool = Field(description="True if this is the lap immediately after a pit stop (out-lap)")
    position: int | None = Field(description="Track position at end of this lap. None if not recorded.")


class DriverLapHistory(BaseModel):
    """All laps for one driver in one session, up to a chosen current lap."""

    driver_code: str = Field(description="3-letter driver code, e.g. 'LEC'")
    team: str = Field(description="Team name, e.g. 'Ferrari'")
    laps_completed: int = Field(description="Number of laps in this response")
    laps: list[LapRecord] = Field(description="Per-lap records in order from lap 1 to current_lap")

class StintRecord(BaseModel):
    """One stint for one driver: a continuous run of laps on one set of tyres."""

    stint_number: int = Field(description="1-indexed stint number for this driver. Stint 1 is from race start to first pit.")
    compound: str = Field(description="Tyre compound used: SOFT, MEDIUM, HARD, INTERMEDIATE, WET")
    start_lap: int = Field(description="Lap number this stint started on (out-lap, or lap 1 for the opening stint)")
    end_lap: int = Field(description="Lap number this stint ended on (in-lap, or current_lap for the ongoing stint)")
    laps_completed: int = Field(description="Number of timed laps in this stint within current_lap window")
    is_ongoing: bool = Field(description="True if this is the driver's current stint (no pit stop has ended it within current_lap)")
    average_lap_time_seconds: float | None = Field(description="Mean lap time across the stint, in seconds. None if no timed laps.")
    best_lap_time_seconds: float | None = Field(description="Fastest lap time in the stint, in seconds. None if no timed laps.")


class DriverStints(BaseModel):
    """All stints for one driver, up to current_lap."""

    driver_code: str = Field(description="3-letter driver code, e.g. 'LEC'")
    team: str = Field(description="Team name, e.g. 'Ferrari'")
    total_stints: int = Field(description="Number of stints in this response")
    current_stint_number: int = Field(description="Stint number the driver is currently on at current_lap")
    stints: list[StintRecord] = Field(description="Stints in chronological order, from race start to current stint")