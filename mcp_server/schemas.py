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

class RivalGap(BaseModel):
    """Gap from the focal driver to one rival, at a specific lap."""

    rival_driver_code: str = Field(description="3-letter code of the rival driver")
    rival_team: str = Field(description="Rival's team name")
    rival_position: int = Field(description="Rival's track position at this lap")
    gap_seconds: float | None = Field(
        description=(
            "Time gap in seconds. Positive means the rival is AHEAD on track "
            "(focal driver crossed the line that many seconds later). Negative "
            "means the rival is behind. None if rivals are on different laps."
        )
    )
    same_lap: bool = Field(description="True if focal driver and rival are on the same lap")

class GapsSnapshot(BaseModel):
    """Full gap picture for one focal driver at one lap."""

    focal_driver_code: str = Field(description="The driver these gaps are computed from")
    focal_position: int = Field(description="Focal driver's track position at this lap")
    current_lap: int = Field(description="The lap these gaps are evaluated at")
    gap_ahead_seconds: float | None = Field(
        description="Gap to the car directly ahead, in seconds. None if leading."
    )
    gap_behind_seconds: float | None = Field(
        description="Gap to the car directly behind, in seconds. None if last."
    )
    gap_to_leader_seconds: float = Field(
        description="Gap to the race leader, in seconds. 0.0 if focal driver IS the leader."
    )
    rivals: list[RivalGap] = Field(
        description="Signed gaps to every other classified driver in the race, sorted by track position"
    )

class WeatherSample(BaseModel):
    """One weather observation at one timestamp."""

    seconds_into_session: float = Field(description="Time elapsed from session start to this observation (in seconds)")
    air_temp_celsius: float = Field(description="Air temperature in degrees Celsius")
    track_temp_celsius: float = Field(description="Track surface temperature in degrees Celsius (typically higher than air temp)")
    humidity_percent: float = Field(description="Relative humidity, 0-100")
    is_raining: bool = Field(description="True if rainfall was detected at this observation. Note: this is a boolean indicator; intensity is not available publicly.")
    wind_speed_ms: float = Field(description="Wind speed in metres per second")


class WeatherSnapshot(BaseModel):
    """Weather at a chosen lap, with a short recent-history trend."""

    current_lap: int = Field(description="The lap this snapshot is evaluated at")
    seconds_into_session: float = Field(description="Time elapsed from session start to current_lap completion, in seconds")
    air_temp_celsius: float = Field(description="Current air temperature in Celsius")
    track_temp_celsius: float = Field(description="Current track temperature in Celsius")
    humidity_percent: float = Field(description="Current relative humidity, 0-100")
    is_raining: bool = Field(description="True if rain is currently falling")
    wind_speed_ms: float = Field(description="Current wind speed in metres per second")

    track_temp_trend: str = Field(description="Trend of track temperature over recent samples: 'rising', 'falling', or 'stable'")
    rain_trend: str = Field(description="Trend of rainfall: 'starting' (was dry, now raining), 'stopping' (was wet, now dry), 'ongoing-wet', 'ongoing-dry'")

    recent_samples: list[WeatherSample] = Field(description="Recent weather observations in chronological order, up to and including the current sample")

class SafetyCarRate(BaseModel):
    """Historical safety car/Virtual Safety Car rate at one circuit, for a chosen lap window."""

    event: str = Field(description="Event/circuit name as queried")
    lap_window_from: int = Field(description="Start of the lap window (inclusive)")
    lap_window_to: int = Field(description="End of the lap window (inclusive)")
    races_analyzed: int = Field(description="Number of historical races used in this estimate")
    races_with_sc_in_window: int = Field(description="Of those, how many had a full safety car within the window")
    races_with_vsc_in_window: int = Field(description="Of those, how many had a VSC within the window")
    races_with_either_in_window: int = Field(description="Of those, how many had SC OR VSC within the window")
    sc_probability: float = Field(description="Estimated probability (0-1) of a full safety car within the window")
    vsc_probability: float = Field(description="Estimated probability (0-1) of a VSC within the window")
    combined_probability: float = Field(description="Estimated probability (0-1) of either SC or VSC within the window")
    sample_size_warning: str | None = Field(description="Warning if sample size is small (n < 5). None if adequate.")


class LiveRaceState(BaseModel):
    """
    Snapshot of the live race at the current simulated moment.
    """

    current_lap: int = Field(description="Highest lap number reached by any driver")
    track_status: str = Field(description="Current track status: 'green', 'yellow', 'sc', 'vsc', 'red'")
    seconds_into_session: float = Field(description="Simulated time elapsed from session start, in seconds")
    is_safety_car_active: bool = Field(description="True if SC or VSC is currently deployed")
    leader_driver_code: str | None = Field(description="3-letter code of the current leader. None if unknown.")
    air_temp_celsius: float | None = Field(description="Latest air temperature reading. None if no weather data yet.")
    track_temp_celsius: float | None = Field(description="Latest track temperature reading. None if no weather data yet.")
    is_raining: bool | None = Field(description="Whether rain is currently falling. None if no weather data yet.")


class LiveDriverStatus(BaseModel):
    """
    Per-driver live snapshot.
    """

    driver_code: str = Field(description="3-letter driver code")
    current_lap: int = Field(description="Last lap this driver completed")
    position: int | None = Field(description="Current track position. None if unknown.")
    last_lap_time_seconds: float | None = Field(description="Most recent lap time. None if not yet recorded.")
    pit_stop_count: int = Field(description="Total pit stops this driver has made")


class LivePitStop(BaseModel):
    """
    A pit stop event surfaced from recent activity.
    """

    driver_code: str = Field(description="3-letter driver code")
    in_lap: int = Field(description="Lap on which the driver pitted")
    compound_from: str | None = Field(description="Compound coming off. None if unknown.")
    seconds_into_session: float = Field(description="Simulated time of the pit entry")


class RecentPitActivity(BaseModel):
    """
    Pit stops within a recent lap window.
    """

    last_n_laps: int = Field(description="The lap window queried, e.g. 5 = last 5 laps")
    current_lap: int = Field(description="Current race lap at the time of the query")
    pit_count: int = Field(description="Number of pit stops in the window")
    pits: list[LivePitStop] = Field(description="Per-stop details, most recent last")