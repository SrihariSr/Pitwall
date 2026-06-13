"""
Watch a historical race play out, with RaceState tracking alongside.
"""
import asyncio
from events.bus import EventBus
from feed_adapter.replay import replay_session
from race_state.state import RaceState


async def logger(bus: EventBus):
    """Print every event as it arrives."""
    async for event in bus.subscribe():
        t = event.seconds_into_session
        if event.event_type == "LapCompleted":
            print(f"[{t:7.1f}s] L{event.lap_number:>2}  {event.driver_code}  {event.lap_time_seconds:.3f}s  P{event.position}")
        elif event.event_type == "PitStop":
            print(f"[{t:7.1f}s] 🛞 PIT IN — {event.driver_code} on lap {event.in_lap} (off {event.compound_from})")
        elif event.event_type == "TrackStatusChange":
            print(f"[{t:7.1f}s] 🏁 STATUS → {event.new_status.upper()} (lap {event.lap_number})")
        elif event.event_type == "WeatherUpdate":
            rain = "🌧️ " if event.is_raining else "☀️ "
            print(f"[{t:7.1f}s] {rain}weather  air {event.air_temp_celsius}°C  track {event.track_temp_celsius}°C")


async def state_snapshot_printer(state: RaceState, interval_seconds: float = 3.0):
    """Every `interval_seconds` of real time, print a state summary."""
    while True:
        await asyncio.sleep(interval_seconds)
        lap = state.current_lap()
        status = state.track_status()
        weather = state.latest_weather()
        rain = "🌧️ raining" if (weather and weather.is_raining) else "☀️ dry"
        leader = next(iter(state.all_drivers()), None)
        leader_str = f"{leader.driver_code} P1 lap {leader.current_lap}" if leader else "—"
        print(f"\n  ═══ STATE  lap={lap}  status={status}  weather={rain}  leader={leader_str} ═══\n")


async def main():
    bus = EventBus()
    state = RaceState()

    await state.start(bus)
    logger_task = asyncio.create_task(logger(bus))
    snapshot_task = asyncio.create_task(state_snapshot_printer(state, interval_seconds=3.0))

    await asyncio.sleep(0.1)

    await replay_session(
        bus,
        year=2022,
        event="Monaco",
        session_type="R",
        speed=30.0,
        start_lap=15,
        end_lap=25,
    )

    await asyncio.sleep(0.5)

    # Final snapshot.
    print("\n══════════════ FINAL STATE ══════════════")
    print(f"Current lap: {state.current_lap()}")
    print(f"Track status: {state.track_status()}")
    weather = state.latest_weather()
    if weather:
        print(f"Weather: {weather.air_temp_celsius}°C air, {weather.track_temp_celsius}°C track, raining={weather.is_raining}")
    print(f"Recent pits (last 5 laps): {len(state.recent_pits(last_n_laps=5))} stops")
    for p in state.recent_pits(last_n_laps=5):
        print(f"  - {p.driver_code} on lap {p.in_lap} off {p.compound_from}")
    print("\nTop 6:")
    for d in state.all_drivers()[:6]:
        print(f"  P{d.position}  {d.driver_code}  lap {d.current_lap}  last {d.last_lap_time_seconds:.3f}s  pits={d.pit_stop_count}")

    logger_task.cancel()
    snapshot_task.cancel()
    await state.stop()


if __name__ == "__main__":
    asyncio.run(main())