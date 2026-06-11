"""
Watch a historical race play out on the terminal via the event bus.

Demonstrates the replay adapter + bus + a simple logging subscriber.
Run: uv run watch_race.py
"""
import asyncio

from events.bus import EventBus
from feed_adapter.replay import replay_session


async def logger(bus: EventBus):
    """
    Print every event as it arrives, formatted compactly.
    """
    async for event in bus.subscribe():
        t = event.seconds_into_session

        if event.event_type == "LapCompleted":
            print(f"[{t:7.1f}s] L{event.lap_number:>2}  {event.driver_code}  {event.lap_time_seconds:.3f}s  P{event.position}")

        elif event.event_type == "PitStop":
            print(f"[{t:7.1f}s] PIT IN {event.driver_code} on lap {event.in_lap} (off {event.compound_from})")

        elif event.event_type == "TrackStatusChange":
            print(f"[{t:7.1f}s] STATUS: {event.new_status.upper()} (lap {event.lap_number})")

        elif event.event_type == "WeatherUpdate":
            rain = "rainy " if event.is_raining else "sunny "
            print(f"[{t:7.1f}s] {rain}weather  air {event.air_temp_celsius}°C  track {event.track_temp_celsius}°C  hum {event.humidity_percent}%")


async def main():
    bus = EventBus()

    # Start the logger as a background task.
    logger_task = asyncio.create_task(logger(bus))

    # Brief pause so the subscriber is ready before events start firing.
    await asyncio.sleep(0.1)

    # Replay Monaco 2022, laps 15-25, at 30× speed.
    # Roughly 30 seconds of real time for 15 minutes of race.
    await replay_session(
        bus,
        year=2022,
        event="Monaco",
        session_type="R",
        speed=30.0,
        start_lap=15,
        end_lap=25,
    )

    # Let the queue drain.
    await asyncio.sleep(0.5)
    logger_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())