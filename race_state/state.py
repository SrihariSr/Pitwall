"""
In-memory race state for Pitwall.

Subscribes to the event bus and maintains a queryable snapshot of the
current race situation. Other components such as MCP tools, orchestrator and
dashboard read from this state instead of re-deriving from raw data.
"""

import asyncio
import asyncio
from collections import deque
from dataclasses import dataclass, field
from events.bus import EventBus
from events.types import Event, LapCompleted, PitStop, TrackStatusChange, WeatherUpdate

@dataclass
class DriverState:
    """
    Per-driver snapshot
    """
    driver_code: str
    current_lap: int = 0
    last_lap_time_seconds: float | None = None
    position: int | None = None
    laps_completed: int = 0
    pit_stop_count: int = 0

class RaceState:
    """
    Current state of "what's going on in the race right now?"
    """
    def __init__(self, total_laps: int | None = None) -> None:
        self.total_laps = total_laps
        self._drivers: dict[str, DriverState] = {} # Per-driver snapshots keyed by their 3-letter code
        # Race wide fields
        self._current_lap: int = 0
        self._track_status: str = "green"
        self._seconds_into_session: float = 0.0
        # Lastest weather sample
        self._latest_weather: WeatherUpdate | None = None
        
        # Recent history
        self._recent_pits: deque[PitStop] = deque(maxlen=100)
        self._recent_status_changes: deque[TrackStatusChange] = deque(maxlen=50)
        self._recent_weather: deque[WeatherUpdate] = deque(maxlen=50)

        self._consumer_task: asyncio.Task | None = None
    
    async def _consume_events_from_queue(self, queue: asyncio.Queue) -> None:
            while True:
                event = await queue.get()
                self._seconds_into_session = event.seconds_into_session

                if isinstance(event, LapCompleted):
                    self._apply_lap_completed(event)
                elif isinstance(event, PitStop):
                    self._apply_pit_stop(event)
                elif isinstance(event, TrackStatusChange):
                    self._apply_track_status(event)
                elif isinstance(event, WeatherUpdate):
                    self._apply_weather(event)

    async def start(self, bus: EventBus) -> None:
            queue = bus.subscribe_queue()  # synchronous registration
            self._consumer_task = asyncio.create_task(self._consume_events_from_queue(queue))

    async def stop(self) -> None:
        if self._consumer_task is not None:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
    
    def _apply_lap_completed(self, event: LapCompleted) -> None:
        driver = self._drivers.setdefault(event.driver_code, DriverState(driver_code=event.driver_code))
        driver.current_lap = event.lap_number
        driver.last_lap_time_seconds = event.lap_time_seconds
        driver.position = event.position
        driver.laps_completed = event.lap_number

        if event.lap_number > self._current_lap:
            self._current_lap = event.lap_number
    
    def _apply_pit_stop(self, event: PitStop) -> None:
        driver = self._drivers.setdefault(
            event.driver_code,
            DriverState(driver_code=event.driver_code),
        )
        driver.pit_stop_count += 1
        self._recent_pits.append(event)

    def _apply_track_status(self, event: TrackStatusChange) -> None:
        self._track_status = event.new_status
        self._recent_status_changes.append(event)

    def _apply_weather(self, event: WeatherUpdate) -> None:
        self._latest_weather = event
        self._recent_weather.append(event)

    def current_lap(self) -> int:
        return self._current_lap

    def track_status(self) -> str:
        """
        Current track status: 'green', 'yellow', 'sc', 'vsc', 'red'.
        """
        return self._track_status

    def seconds_into_session(self) -> float:
        return self._seconds_into_session

    def driver(self, driver_code: str) -> DriverState | None:
        """
        Per-driver snapshot, or None if we haven't seen this driver yet.
        """
        return self._drivers.get(driver_code)

    def all_drivers(self) -> list[DriverState]:
        """
        All drivers we've seen, sorted by current position (unknowns last).
        """
        drivers = list(self._drivers.values())
        drivers.sort(key=lambda d: d.position if d.position is not None else 99)
        return drivers

    def latest_weather(self) -> WeatherUpdate | None:
        """
        Most recent weather sample, or None if none yet.
        """
        return self._latest_weather

    def recent_pits(self, last_n_laps: int = 5) -> list[PitStop]:
        """
        Pit stops within the most recent n laps.
        """
        cutoff = self._current_lap - last_n_laps
        return [p for p in self._recent_pits if p.in_lap >= cutoff]

    def is_safety_car_active(self) -> bool:
        return self._track_status in ("sc", "vsc")