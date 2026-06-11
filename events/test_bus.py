"""
Sanity test for the event bus.

Spawns a producer that emits 5 events, two subscribers (one for all events,
one filtered to WeatherUpdate only), and verifies the fan-out works.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from events import LapCompleted, WeatherUpdate
from events.bus import EventBus


async def producer(bus: EventBus):
    """Emit a few mixed events with small delays."""
    for lap in range(1, 4):
        await asyncio.sleep(0.5)
        await bus.publish(LapCompleted(
            seconds_into_session=lap * 90.0,
            driver_code="LEC",
            lap_number=lap,
            lap_time_seconds=90.0,
            position=1,
        ))

    await asyncio.sleep(0.5)
    await bus.publish(WeatherUpdate(
        seconds_into_session=300.0,
        air_temp_celsius=22.0,
        track_temp_celsius=27.5,
        humidity_percent=78.0,
        is_raining=True,
        wind_speed_ms=0.5,
    ))

    await asyncio.sleep(0.5)
    await bus.publish(LapCompleted(
        seconds_into_session=360.0,
        driver_code="LEC",
        lap_number=4,
        lap_time_seconds=89.5,
        position=1,
    ))

    # In a real system we'd have a "done" event; for the test we just
    # rely on the consumers being cancelled by main().


async def listen_all(bus: EventBus):
    """Subscriber that receives every event."""
    print("[ALL] subscribed")
    async for event in bus.subscribe():
        print(f"[ALL] received {event.event_type}: lap={getattr(event, 'lap_number', None)}, raining={getattr(event, 'is_raining', None)}")


async def listen_weather(bus: EventBus):
    """Subscriber that receives only weather events."""
    print("[WEATHER] subscribed")
    async for event in bus.subscribe(event_types=["WeatherUpdate"]):
        print(f"[WEATHER] received update: track_temp={event.track_temp_celsius}, raining={event.is_raining}")


async def main():
    bus = EventBus()

    # Start the listeners as background tasks. They run forever; we'll
    # cancel them after the producer finishes.
    listen_all_task = asyncio.create_task(listen_all(bus))
    listen_weather_task = asyncio.create_task(listen_weather(bus))

    # Give listeners a moment to subscribe before producing.
    await asyncio.sleep(0.1)

    # Run the producer to completion.
    await producer(bus)

    # Let the queues drain.
    await asyncio.sleep(0.2)

    # Tidy up.
    listen_all_task.cancel()
    listen_weather_task.cancel()

    print("\nTest finished.")


asyncio.run(main())