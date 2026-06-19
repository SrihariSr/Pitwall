"""
The Weather Watcher subagent.

Monitors current rainfall and track conditions, recent weather trends, and
identifies the optimal tyre compound family. Reports whether a compound
pivot may be needed soon. The Orchestrator combines this with the Tyre
Strategist's view of the current stint to decide whether to pit for a
compound change.
"""

from requests_cache.models import response
import os
from mcp_server.live_state import get_active_state
from mcp_server.server import get_weather, get_current_race_state
from llm.client import LLMClient
from agents.schemas import WeatherAssessment

DEFAULT_MODEL = os.getenv("LLM_MODEL_SUBAGENT", "gemini-2.5-flash-lite")

SYSTEM_PROMPT = """
You are the Weather Watcher on an F1 team's pit wall.

Your only job: monitor weather and track conditions, identify the optimal tyre compound family for the current state of the track, and flag when a compound pivot may be needed soon.

You DO NOT care about:
- Tyre degradation within a stint (the Tyre Strategist handles that, you only assess fit-for-conditions)
- Gap dynamics (the Gap Analyst handles that)
- Whether to pit (the Orchestrator decides)
- Safety car probability (the Safety Car Oracle handles that)

You DO care about:
- Current rainfall: dry, light rain, heavy rain
- Track surface state inferred from rainfall trend and temperatures: dry, damp, fully wet
- The direction of change: is it drying out or getting wetter?
- Track temperature: high track temps accelerate drying after rainfall stops

How to think:
- Dry conditions + stable temps: slicks are correct, no pivot expected.
- Light rain on a dry track: inters become optimal; do not pivot on a single drop, look for a sustained trend.
- Heavy rain on any track: wets needed.
- Rain just stopped + track drying: inters now, slicks within 5-10 laps if temperatures support it.
- Drying track with warm asphalt (>35°C track temp): dry line emerges quickly after rain stops.
- Drying track with cool asphalt (<25°C track temp): track stays damp for longer.

Compound categories:
- "slicks": dry track, no rainfall
- "inters": light rain, damp track, transitional periods
- "wets": heavy standing water

Pivot urgency calibration:
- "immediate": conditions clearly warrant a pivot RIGHT NOW (e.g. heavy rain on slicks, or fully dry track on wets)
- "soon": optimal compound likely to change within 3-5 laps (e.g. drying track will need slicks soon)
- "stable": current compound family fits conditions and no change expected in the next 5+ laps

Biases:
- Don't pivot on a single sample. Look for trend across multiple data points.
- Be honest about uncertainty during transitions, confidence should drop when conditions are volatile.
- Cite specific signals in reasoning (e.g. "track temp risen 4C in last 8 minutes", not "conditions improving").
"""

def _build_user_prompt(current_lap, race, weather) -> str:
    """
    Composes the per-call part of the prompt. Shows current snapshot + recent weather samples.
    """
    samples = weather.recent_samples[-8:] # Last 8 minutes of samples

    weather_lines = "\n".join(
        f"t={s.seconds_into_session:.0f}s: "
        f"air {s.air_temp_celsius:.1f}°C, track {s.track_temp_celsius:.1f}°C, "
        f"humidity {s.humidity_percent:.0f}%, "
        f"{'RAIN' if s.is_raining else 'dry'}"
        for s in samples
        )
    
    air_temp_str = f"{race.air_temp_celsius}°C" if getattr(race, 'air_temp_celsius', None) else "unknown"
    track_temp_str = f"{race.track_temp_celsius}°C" if race.track_temp_celsius else "unknown"

    return f"""
    Race at lap {current_lap}.
    Current snapshot:
      Track status: {race.track_status}
      Currently: {'RAINING' if race.is_raining else 'dry'}
      Track temp: {track_temp_str}
      Air temp:   {air_temp_str}

    Recent weather samples (most recent last):
    {weather_lines}

    Assess the current condition, optimal compound, pivot urgency, and your reasoning.
    """

async def assess_weather(
    client: LLMClient,
    year: int,
    event: str,
    session_type: str,
    model: str = DEFAULT_MODEL
) -> WeatherAssessment:
    """
    Estimate current weather state and optimal compound family.

    Parameters:
        client: shared LLM client
        year, event, session_type: identify the session for MCP queries
        model: which LLM to use. Default reads LLM_MODEL_SUBAGENT from .env.
    """
    state = get_active_state()
    if state is None:
        raise RuntimeError("No live RaceState active! Cannot access weather.")
    
    current_lap = state.current_lap()
    if current_lap < 1:
        raise RuntimeError("Race has not started yet.")
    
    race = get_current_race_state()
    
    weather = get_weather(
        year=year,
        event=event,
        session_type=session_type,
        current_lap=current_lap
    )

    user_prompt = _build_user_prompt(current_lap, race, weather)

    return await client.generate_structured(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_schema=WeatherAssessment,
        temperature=0.2
    )