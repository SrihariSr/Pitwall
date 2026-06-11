"""
Replay adapter: load a historical FastF1 session and stream events onto an EventBus.

Events are emitted in session-time order with real-time delays scaled by `speed`.
At speed=30, 30 seconds of simulated race time plays back in 1 real second.
"""
import asyncio
import pandas as pd

from events.bus import EventBus
from events.types import Event, LapCompleted, PitStop, TrackStatusChange, WeatherUpdate
from mcp_server.sessions import load_session

_STATUS_MAP = {
    "1": "green",
    "2": "yellow",
    "4": "sc",
    "5": "red",
    "6": "vsc",
    "7": "vsc",
}


def _dominant_status(raw: str) -> str:
    """Pick the most severe single-flag character from a multi-flag string like '14'."""
    chars = [c for c in raw if c.isdigit()]
    if not chars:
        return "green"
    best = max(chars, key=lambda c: int(c))
    return _STATUS_MAP.get(best, "green")


def _build_events(session, start_lap: int, end_lap: int) -> list[tuple[float, Event]]:
    laps = session.laps
    events: list[tuple[float, Event]] = []

    # --- LapCompleted ---
    valid = laps[
        laps["LapNumber"].between(start_lap, end_lap)
        & laps["LapTime"].notna()
        & laps["Time"].notna()
    ]
    for _, row in valid.iterrows():
        t = float(row["Time"].total_seconds())
        events.append((t, LapCompleted(
            seconds_into_session=t,
            driver_code=str(row["Driver"]),
            lap_number=int(row["LapNumber"]),
            lap_time_seconds=float(row["LapTime"].total_seconds()),
            position=int(row["Position"]) if pd.notna(row["Position"]) else None,
        )))

    # --- PitStop ---
    pit_rows = laps[
        laps["LapNumber"].between(start_lap, end_lap)
        & laps["PitInTime"].notna()
    ]
    for _, row in pit_rows.iterrows():
        pit_t = float(row["PitInTime"].total_seconds())
        driver = str(row["Driver"])
        in_lap = int(row["LapNumber"])

        next_laps = laps[(laps["Driver"] == driver) & (laps["LapNumber"] == in_lap + 1)]
        out_lap = int(next_laps.iloc[0]["LapNumber"]) if len(next_laps) > 0 else None
        compound_to = (
            str(next_laps.iloc[0]["Compound"])
            if len(next_laps) > 0 and pd.notna(next_laps.iloc[0]["Compound"])
            else None
        )

        events.append((pit_t, PitStop(
            seconds_into_session=pit_t,
            driver_code=driver,
            in_lap=in_lap,
            out_lap=out_lap,
            compound_from=str(row["Compound"]) if pd.notna(row["Compound"]) else None,
            compound_to=compound_to,
        )))

    # --- TrackStatusChange ---
    # One row per lap (first driver with a valid lap time); emit when status changes.
    per_lap = (
        laps[
            laps["LapNumber"].between(start_lap, end_lap)
            & laps["LapTime"].notna()
            & laps["Time"].notna()
        ]
        .sort_values("LapNumber")
        .drop_duplicates(subset="LapNumber", keep="first")
    )
    prev_status = None
    for _, row in per_lap.iterrows():
        raw = str(row["TrackStatus"]) if pd.notna(row["TrackStatus"]) else "1"
        status = _dominant_status(raw)
        if status != prev_status:
            lap_start_t = float(row["Time"].total_seconds()) - float(row["LapTime"].total_seconds())
            events.append((lap_start_t, TrackStatusChange(
                seconds_into_session=lap_start_t,
                lap_number=int(row["LapNumber"]),
                new_status=status,
            )))
            prev_status = status

    # --- WeatherUpdate ---
    weather = session.weather_data.copy()
    weather["_t"] = weather["Time"].dt.total_seconds()

    window_laps = laps[
        laps["LapNumber"].between(start_lap, end_lap)
        & laps["Time"].notna()
        & laps["LapTime"].notna()
    ]
    if len(window_laps) > 0:
        w_start = float((window_laps["Time"] - window_laps["LapTime"]).dt.total_seconds().min())
        w_end = float(window_laps["Time"].dt.total_seconds().max())
        weather = weather[weather["_t"].between(w_start, w_end)]

    for _, row in weather.iterrows():
        t = float(row["_t"])
        events.append((t, WeatherUpdate(
            seconds_into_session=t,
            air_temp_celsius=float(row["AirTemp"]),
            track_temp_celsius=float(row["TrackTemp"]),
            humidity_percent=float(row["Humidity"]),
            is_raining=bool(row["Rainfall"]),
            wind_speed_ms=float(row["WindSpeed"]),
        )))

    events.sort(key=lambda x: x[0])
    return events


async def replay_session(
    bus: EventBus,
    year: int,
    event: str,
    session_type: str,
    speed: float = 1.0,
    start_lap: int | None = None,
    end_lap: int | None = None,
) -> None:
    """
    Load a historical session and publish its events to `bus` in real time.

    Parameters:
        bus: EventBus to publish onto.
        year: Race year, e.g. 2022.
        event: Event name like "Monaco" or round number as string.
        session_type: "R", "Q", "FP1", "FP2", "FP3", "S".
        speed: Playback speed multiplier. 1.0 = real time, 30.0 = 30× faster.
        start_lap: First lap to include (default: 1).
        end_lap: Last lap to include (default: final lap of session).
    """
    session = load_session(year, event, session_type)
    laps = session.laps

    max_lap = int(laps["LapNumber"].max())
    if start_lap is None:
        start_lap = 1
    if end_lap is None:
        end_lap = max_lap

    events = _build_events(session, start_lap, end_lap)
    if not events:
        return

    ref_session_time = events[0][0]
    replay_start = asyncio.get_event_loop().time()

    for session_time, ev in events:
        target = replay_start + (session_time - ref_session_time) / speed
        delay = target - asyncio.get_event_loop().time()
        if delay > 0:
            await asyncio.sleep(delay)
        await bus.publish(ev)
