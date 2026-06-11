"""
Exposes get_session_state, which returns a high-level snapshot of an
F1 session at a given lap. Designed for replay mode: we operate on
historical sessions and pretend we're at a chosen lap.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.fastmcp import FastMCP
import pandas as pd

from mcp_server.sessions import load_session
from mcp_server.schemas import SessionState, DriverLapHistory, LapRecord, DriverStints, StintRecord, RivalGap, GapsSnapshot, WeatherSnapshot, WeatherSample, SafetyCarRate

mcp = FastMCP("pitwall")

# Mapping from FastF1's track-status numeric codes to our human-readable labels.
_TRACK_STATUS_MAP = {
    "1": "green",
    "2": "yellow",
    "4": "sc",    # safety car
    "5": "red",
    "6": "vsc",   # virtual safety car deployed
    "7": "vsc",   # virtual safety car ending
}

@mcp.tool()
def get_session_state(
    year: int,
    event: str,
    session_type: str,
    current_lap: int,
) -> SessionState:
    """
    Return a high-level snapshot of an F1 session at a chosen lap.

    Use this when you need orientation: what session, what lap, who is
    leading, what the track status is. This is the first call most agents
    should make to know where they are in the race.

    Parameters:
        year: Race year, e.g. 2022
        event: Event name like "Monaco" or "Hungarian Grand Prix"
        session_type: "R" for race, "Q" for qualifying, "FP1"/"FP2"/"FP3", "S" for sprint
        current_lap: Lap number to evaluate state at. Must be >= 1.
    """
    session = load_session(year, event, session_type)

    laps = session.laps
    total_laps = int(laps["LapNumber"].max())

    # Clamp to the actual race length to avoid silent confusion.
    if current_lap < 1 or current_lap > total_laps:
        raise ValueError(f"current_lap {current_lap} out of range for this session!")

    # Find the leader at current_lap. The leader is the driver with Position == 1 on that lap. 
    # If multiple rows match (rare case), we take the first.
    lap_rows = laps[laps["LapNumber"] == current_lap]
    leader_rows = lap_rows[lap_rows["Position"] == 1]

    if len(leader_rows) == 0:
        #fall back to whoever has the fastest lap so far if data is missing for this lap
        leader_code = "UNK"
        leader_time = None
    else:
        leader = leader_rows.iloc[0]
        leader_code = str(leader["Driver"])
        leader_time = (
            float(leader["LapTime"].total_seconds())
            if pd.notna(leader["LapTime"])
            else None
        )
    # Track status at this lap. FastF1 stores per-lap status in the laps DataFrame.
    raw_status = str(lap_rows["TrackStatus"].iloc[0]) if len(lap_rows) > 0 else "1"
    
    # TrackStatus can be a multi-flag string like "14" (yellow + SC); we take the
    # most-severe single character for simplicity.
    status_char = max(raw_status, key=lambda c: int(c)) if raw_status else "1"
    track_status = _TRACK_STATUS_MAP.get(status_char, "green")

    return SessionState(
        year=year,
        event=session.event["EventName"],
        session_type=session_type,
        total_laps=total_laps,
        current_lap=current_lap,
        leader_driver_code=leader_code,
        leader_lap_time_seconds=leader_time,
        track_status=track_status,
    )


@mcp.tool()
def get_driver_lap_history(
    year: int,
    event: str,
    session_type: str,
    driver_code: str,
    current_lap: int,
) -> DriverLapHistory:
    """
    Return a driver's per-lap history from lap 1 up to current_lap (inclusive).

    Use this to analyse stint pace, fit tyre degradation curves, detect
    cliffs, compute rolling averages, or compare a driver's pace lap-by-lap.
    The data is truncated at current_lap — the tool will never return laps
    that haven't yet happened in the simulation, even though they exist
    in the underlying historical data.

    Laps with missing lap times (formation lap, timing dropouts) are
    excluded from the response.

    Parameters:
        year: Race year, e.g. 2022
        event: Event name like "Monaco" or round number as string
        session_type: "R", "Q", "FP1", "FP2", "FP3", "S"
        driver_code: 3-letter driver code, e.g. "PIA", "LEC"
        current_lap: Truncate history at this lap (inclusive). Must be >= 1.
    """
    session = load_session(year, event, session_type)

    # Filter to this driver, then truncate to laps already completed.
    driver_laps = session.laps.pick_drivers(driver_code)
    driver_laps = driver_laps[driver_laps["LapNumber"] <= current_lap]

    # Drop laps with no recorded lap time. NaT (Not-a-Time) shows up for
    # the formation lap and any timing dropouts.
    driver_laps = driver_laps[driver_laps["LapTime"].notna()]

    if len(driver_laps) == 0:
        raise ValueError(
            f"No lap data for driver {driver_code} in {year} {event} {session_type} "
            f"up to lap {current_lap}. Check the driver code and that the session has started."
        )

    # Pull team name from the first row — it's the same on every lap.
    team = str(driver_laps.iloc[0]["Team"])

    # Build the LapRecord list. We iterate row-by-row because pandas columns
    # need conversion to native Python types for Pydantic.
    records: list[LapRecord] = []
    for _, row in driver_laps.iterrows():
        records.append(LapRecord(
            lap_number=int(row["LapNumber"]),
            lap_time_seconds=float(row["LapTime"].total_seconds()),
            sector_1_seconds=_timedelta_to_seconds(row.get("Sector1Time")),
            sector_2_seconds=_timedelta_to_seconds(row.get("Sector2Time")),
            sector_3_seconds=_timedelta_to_seconds(row.get("Sector3Time")),
            compound=str(row["Compound"]) if pd.notna(row["Compound"]) else "UNKNOWN",
            tyre_life=int(row["TyreLife"]) if pd.notna(row["TyreLife"]) else 0,
            stint=int(row["Stint"]) if pd.notna(row["Stint"]) else 1,
            is_pit_in_lap=pd.notna(row["PitInTime"]),
            is_pit_out_lap=pd.notna(row["PitOutTime"]),
            position=int(row["Position"]) if pd.notna(row["Position"]) else None,
        ))

    return DriverLapHistory(
        driver_code=driver_code,
        team=team,
        laps_completed=len(records),
        laps=records,
    )

def _timedelta_to_seconds(value) -> float | None:
    """
    Convert a pandas Timedelta to seconds, or None if missing.

    Sector times come through as Timedelta objects when present and
    NaT (Not-a-Time) when not. Pydantic needs a float or None.
    """
    if value is None or pd.isna(value):
        return None
    return float(value.total_seconds())

@mcp.tool()
def get_tyre_stints(
    year: int,
    event: str,
    session_type: str,
    driver_code: str,
    current_lap: int,
) -> DriverStints:
    """
    Return the stint history of one driver upto the current lap

    Each stint represents a continuous run of laps on one set of tyres.
    Use this when you need a stint-level view rather than per-lap detail —
    for example, to ask 'what compound is this driver on?', 'how old are
    their current tyres?', 'how did their last stint pace compare to their
    rivals' equivalent stints?'.

    The driver's current (ongoing) stint is included with is_ongoing=True
    and end_lap set to current_lap. The stint's true end is unknown until
    the driver pits.

    Parameters:
        year: Race year, e.g. 2022
        event: Event name like "Monaco" or a round number as a string
        session_type: "R", "Q", "FP1", "FP2", "FP3", "S"
        driver_code: 3-letter driver code, e.g. "PIA", "LEC"
        current_lap: Lap to truncate at (inclusive), must be >= 1
    """
    session = load_session(year, event, session_type)

    driver_laps = session.laps.pick_drivers(driver_code)
    driver_laps = driver_laps[driver_laps["LapNumber"] <= current_lap]
    driver_laps = driver_laps[driver_laps["LapTime"].notna()]

    if len(driver_laps) == 0:
        raise ValueError(f"No lap data for {driver_code} the {year} {event} {session_type} upto lap {current_lap}")
    
    team = str(driver_laps.iloc[0]["Team"])

    # Gets the highest stint number a driver has reached by current_lap
    max_stint = int(driver_laps["Stint"].max())

    stints: list[StintRecord] = []
    for stint_number in range(1, max_stint + 1):
        stint_laps = driver_laps[driver_laps["Stint"] == stint_number]
        if len(stint_laps) == 0:
            continue

        start_lap = int(stint_laps["LapNumber"].min())
        end_lap = int(stint_laps["LapNumber"].max())

        compound = str(stint_laps["Compound"].iloc[0]) if pd.notna(stint_laps["Compound"].iloc[0]) else "UNKNOWN"

        lap_times_seconds = [t.total_seconds() for t in stint_laps["LapTime"] if pd.notna(t)]
        avg_lap = float(sum(lap_times_seconds) / len(lap_times_seconds)) if lap_times_seconds else None
        best_lap = float(min(lap_times_seconds)) if lap_times_seconds else None
        last_row = stint_laps.iloc[-1]
        pitted_on_last_lap = pd.notna(last_row["PitInTime"])
        is_ongoing = (stint_number == max_stint) and (not pitted_on_last_lap)

        stints.append(StintRecord(
            stint_number=stint_number,
            compound=compound,
            start_lap=start_lap,
            end_lap=end_lap,
            laps_completed=len(stint_laps),
            is_ongoing=is_ongoing,
            average_lap_time_seconds=avg_lap,
            best_lap_time_seconds=best_lap,
        ))

    
    return DriverStints(
        driver_code=driver_code,
        team=team,
        total_stints=len(stints),
        current_stint_number=max_stint,
        stints=stints,
    )

def _get_lap_end_times(session, lap_number: int) -> dict[str, dict]:
    """
    Return a mapping of driver_code -> {time_seconds, position, team, lap_number}
    for every driver who has data at the given lap.

    time_seconds is the cumulative race time (seconds from session start) when
    the driver crossed the line completing lap_number.

    Drivers on a different lap (lapped down) are returned with their actual most
    recent lap_number, so the caller can detect lap mismatches.
    """

    laps = session.laps
    result: dict[str, dict] = {}

    for driver_code in laps["Driver"].unique():
        driver_laps = laps[laps["Driver"] == driver_code]
        driver_laps = driver_laps[driver_laps["LapNumber"] <= lap_number]
        if len(driver_laps) == 0:
            continue

        latest = driver_laps.sort_values("LapNumber").iloc[-1]
        if pd.isna(latest["Time"]):
            continue

        result[str(driver_code)] = {
            "time_seconds": float(latest["Time"].total_seconds()),
            "position": int(latest["Position"]) if pd.notna(latest["Position"]) else 99,
            "team": str(latest["Team"]),
            "lap_number": int(latest["LapNumber"]),
        }

    return result


@mcp.tool()
def get_gaps_to_rivals(
    year: int,
    event: str,
    session_type: str,
    driver_code: str,
    current_lap: int,
) -> GapsSnapshot:
    """
    Return time gaps from one focal driver to every other driver at a chosen lap.

    Use this to evaluate strategic position: who is in undercut range, who is
    threatening from behind, how far back from the leader is the focal driver.
    Sign convention: positive gap means the rival is ahead on track (focal
    driver crossed the line that many seconds later). Negative means behind.
    Lapped drivers appear with gap_seconds=None.

    Parameters:
        year: Race year, e.g. 2022
        event: Event name like "Monaco" or round number as string
        session_type: "R", "Q", "FP1", "FP2", "FP3", "S"
        driver_code: 3-letter focal driver code, e.g. "LEC"
        current_lap: Lap to evaluate gaps at. Must be >= 1.
    """
    session = load_session(year, event, session_type)

    times_by_driver = _get_lap_end_times(session, current_lap)

    if driver_code not in times_by_driver:
        raise ValueError(f"Driver {driver_code} has no data at lap {current_lap} in {year} {event} {session_type}. Check the code and lap range!")

    focal = times_by_driver[driver_code]
    focal_lap = focal["lap_number"]
    focal_time = focal["time_seconds"]
    focal_position = focal["position"]

    rivals: list[RivalGap] = []

    for code, info in times_by_driver.items():
        if code == driver_code:
            continue
        same_lap = info["lap_number"] == focal_lap

        gap_seconds = focal_time - info["time_seconds"] if same_lap else None

        rivals.append(
            RivalGap(rival_driver_code=code, rival_team=info["team"], rival_position=info["position"], gap_seconds=gap_seconds, same_lap=same_lap)
        )

    rivals.sort(key=lambda r: r.rival_position)

    rivals_ahead_same_lap = [r for r in rivals if r.same_lap and r.rival_position < focal_position]
    rivals_behind_same_lap = [r for r in rivals if r.same_lap and r.rival_position > focal_position]

    gap_ahead = min((r.gap_seconds for r in rivals_ahead_same_lap), default=None)
    signed_gap_behind = max((r.gap_seconds for r in rivals_behind_same_lap), default=None)

    if focal_position == 1:
        gap_to_leader = 0.0
    else:
        leader = next((r for r in rivals if r.rival_position == 1 and r.same_lap), None)
        gap_to_leader = leader.gap_seconds if leader is not None else None

    return GapsSnapshot(
        focal_driver_code=driver_code,
        focal_position=focal_position,
        current_lap=current_lap,
        gap_ahead_seconds=gap_ahead,
        gap_behind_seconds=abs(signed_gap_behind) if signed_gap_behind is not None else None,
        gap_to_leader_seconds=abs(gap_to_leader) if gap_to_leader is not None else None,
        rivals=rivals,
    )

def _session_time_at_lap(session, lap_number: int) -> float | None:
    """
    Return the cumulative session time (in seconds) when lap_number was completed.

    Defined as: the time at which the LEADER of lap_number crossed the line.
    Returns None if the lap has no recorded time data.
    """
    laps = session.laps
    lap_rows = laps[laps["LapNumber"] == lap_number]
    lap_rows = lap_rows[lap_rows["Time"].notna()]
    
    if len(lap_rows) == 0:
        return None
    
    leader_time = lap_rows["Time"].min()
    return float(leader_time.total_seconds())

@mcp.tool()
def get_weather(
    year: int,
    event: str,
    session_type: str,
    current_lap: int,
    history_samples: int = 10,
) -> WeatherSnapshot:
    """
    Return current weather conditions at a chosen lap, plus recent trend.

    Weather data is sampled approximately every minute throughout the session,
    independent of lap structure. This tool truncates to observations taken
    before the leader completed current_lap, so no future-leaking.

    The recent_samples list gives short-term history (default last 10 samples,
    roughly the last 10 minutes) so reasoning agents can detect trends —
    rising track temp, incoming rain, dropping humidity.

    Note: rainfall is reported as a boolean only. The public feed does not
    expose intensity (light drizzle vs heavy rain). Combine with humidity
    and track temperature to infer intensity if needed.

    Parameters:
        year: Race year, e.g. 2022
        event: Event name like "Monaco" or round number as string
        session_type: "R", "Q", "FP1", "FP2", "FP3", "S"
        current_lap: Lap to evaluate weather at. Must be >= 1.
        history_samples: How many recent samples to return for trend analysis. Default 10.
    """
    session = load_session(year, event, session_type)
    target_seconds = _session_time_at_lap(session, current_lap)

    if target_seconds is None:
        raise ValueError(
            f"No lap-time data at lap {current_lap} for {year} {event} {session_type}. Cannot respond to weather query to this lap."
        )
    
    weather = session.weather_data.copy()
    # Converting time column to seconds
    weather["seconds_into_session"] = weather["Time"].dt.total_seconds()
    weather = weather[weather["seconds_into_session"] <= target_seconds]

    if len(weather) == 0:
        raise ValueError(
            f"No weather samples available before lap {current_lap}. The session has only just started."
        )

    weather = weather.sort_values("seconds_into_session")
    
    recent = weather.tail(history_samples)
    current_row = recent.iloc[-1]

    # Build WeatherSample list. We round to one decimal for readability.
    def _row_to_sample(row) -> WeatherSample:
        return WeatherSample(
            seconds_into_session=float(row["seconds_into_session"]),
            air_temp_celsius=round(float(row["AirTemp"]), 1),
            track_temp_celsius=round(float(row["TrackTemp"]), 1),
            humidity_percent=round(float(row["Humidity"]), 1),
            is_raining=bool(row["Rainfall"]),
            wind_speed_ms=round(float(row["WindSpeed"]), 1),
        )

    recent_samples = [_row_to_sample(r) for _, r in recent.iterrows()]

    # Compute track temp trend. Compare oldest vs newest in the window.
    if len(recent_samples) >= 2:
        delta = recent_samples[-1].track_temp_celsius - recent_samples[0].track_temp_celsius
        if delta > 1.0:
            track_temp_trend = "rising"
        elif delta < -1.0:
            track_temp_trend = "falling"
        else:
            track_temp_trend = "stable"
    else:
        track_temp_trend = "stable"

    # Compute rain trend: compare current rain state to earliest in window.
    if len(recent_samples) >= 2:
        was_raining = recent_samples[0].is_raining
        is_raining_now = recent_samples[-1].is_raining
        if not was_raining and is_raining_now:
            rain_trend = "starting"
        elif was_raining and not is_raining_now:
            rain_trend = "stopping"
        elif is_raining_now:
            rain_trend = "ongoing-wet"
        else:
            rain_trend = "ongoing-dry"
    else:
        rain_trend = "ongoing-wet" if recent_samples[0].is_raining else "ongoing-dry"

    return WeatherSnapshot(
        current_lap=current_lap,
        seconds_into_session=float(current_row["seconds_into_session"]),
        air_temp_celsius=round(float(current_row["AirTemp"]), 1),
        track_temp_celsius=round(float(current_row["TrackTemp"]), 1),
        humidity_percent=round(float(current_row["Humidity"]), 1),
        is_raining=bool(current_row["Rainfall"]),
        wind_speed_ms=round(float(current_row["WindSpeed"]), 1),
        track_temp_trend=track_temp_trend,
        rain_trend=rain_trend,
        recent_samples=recent_samples,
    )    

from functools import lru_cache


def _find_safety_car_laps(session) -> tuple[set[int], set[int]]:
    """
    Inspect one session and return (sc_laps, vsc_laps).

    Each is a set of lap numbers during which the respective condition was
    active. TrackStatus codes per FastF1: '4' = SC, '6'/'7' = Virtual Safety Car deployed/ending.
    A multi-flag string like '46' means both were active that lap.
    """
    sc_laps: set[int] = set()
    vsc_laps: set[int] = set()

    for _, row in session.laps.iterrows():
        if pd.isna(row["LapNumber"]) or pd.isna(row["TrackStatus"]):
            continue
        status = str(row["TrackStatus"])
        lap = int(row["LapNumber"])
        if "4" in status:
            sc_laps.add(lap)
        if "6" in status or "7" in status:
            vsc_laps.add(lap)

    return sc_laps, vsc_laps


@lru_cache(maxsize=32)
def _historical_sc_data(event: str, years_tuple: tuple[int, ...]) -> tuple:
    """
    Load and aggregate Safety Car/Virtual Safety Car data for one circuit across multiple years.

    Cached by (event, years_tuple): the first call loads N sessions, subsequent calls return instantly.

    Returns a tuple of (year, total_laps, sc_laps_frozenset, vsc_laps_frozenset)
    for each year that loaded successfully. Years missing or with errors are skipped.
    """
    out = []
    for year in years_tuple:
        try:
            session = load_session(year, event, "R")
            total_laps = int(session.laps["LapNumber"].max())
            sc_laps, vsc_laps = _find_safety_car_laps(session)
            out.append((year, total_laps, frozenset(sc_laps), frozenset(vsc_laps)))
        except Exception:
            continue  # Race may not have existed that year. Skip and continue.
    return tuple(out)

@mcp.tool()
def historical_sc_rate(
    event: str,
    lap_from: int,
    lap_to: int,
    years_back: int = 8
) -> SafetyCarRate:
    """
    Estimate the historical probability of a safety car or Virtual Safety Car at this circuit
    within a chosen lap window, based on past races at the same venue.

    This is a coarse statistical prior, not a real-time prediction. Treat it as
    base-rate information to combine with current race context (incidents,
    weather, driver behaviour). Sample size is small (~8 races per circuit
    in the reliable FastF1 era from 2018 onward), so rates are noisy estimates.

    Parameters:
        event: Event name like "Monaco" or "British Grand Prix"
        lap_from: Start of the lap window (inclusive, >= 1)
        lap_to: End of the lap window (inclusive, >= lap_from)
        years_back: How many years of history to consider. Default 8 (full reliable era).
    """


    if lap_from < 1 or lap_to < lap_from:
        raise ValueError(f"Invalid lap window: {lap_from}..{lap_to}")

    # Build year range. End on the last completed season; start years_back behind that.
    end_year = 2025  # last full season as of project start
    start_year = max(2018, end_year - years_back + 1)
    years = tuple(range(start_year, end_year + 1))

    historical = _historical_sc_data(event, years)

    if len(historical) == 0:
        raise ValueError(
            f"No historical data loaded for event '{event}'. Check spelling! FastF1 expects names like 'Monaco' or 'British Grand Prix'."
        )

    races_with_sc = 0
    races_with_vsc = 0
    races_with_either = 0

    for (_year, total_laps, sc_laps, vsc_laps) in historical:
        # If the window starts beyond this race's length, skip it; the window is irrelevant for shorter historical editions of the race.
        if lap_from > total_laps:
            continue

        has_sc = any(lap_from <= L <= lap_to for L in sc_laps)
        has_vsc = any(lap_from <= L <= lap_to for L in vsc_laps)

        if has_sc:
            races_with_sc += 1
        if has_vsc:
            races_with_vsc += 1
        if has_sc or has_vsc:
            races_with_either += 1

    n = len(historical)
    warning = None
    if n < 5:
        warning = f"Only {n} historical races available; rate estimates are noisy with this small sample size."
    
    return SafetyCarRate(
        event=event,
        lap_window_from=lap_from,
        lap_window_to=lap_to,
        races_analyzed=n,
        races_with_sc_in_window=races_with_sc,
        races_with_vsc_in_window=races_with_vsc,
        races_with_either_in_window=races_with_either,
        sc_probability=races_with_sc / n,
        vsc_probability=races_with_vsc / n,
        combined_probability=races_with_either / n,
        sample_size_warning=warning,
    )

if __name__ == "__main__":
    mcp.run()