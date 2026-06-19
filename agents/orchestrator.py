"""
The Orchestrator: fuses subagent outputs into a single pit-stop call.

Wakes selectively (every N laps + on triggers), calls subagents in parallel,
asks a stronger model to fuse their structured outputs into a PitDecision.
Logs every decision to decisions.jsonl for post-race auditing.
"""
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal
from mcp_server.live_state import get_active_state
from mcp_server.server import get_current_race_state
from llm.client import LLMClient
from agents.schemas import (
    PitDecision,
    TyreAssessment,
    GapAssessment,
    MonteCarloAssessment,
)
from agents.tyre_strategist import assess_tyres
from agents.gap_analyst import assess_gaps
from agents.monte_carlo import assess_monte_carlo

ORCHESTRATOR_MODEL = os.getenv("LLM_MODEL_ORCHESTRATOR", "gemini-2.5-flash")

_DECISIONS_PATH = Path("decisions/decisions.jsonl")

SYSTEM_PROMPT = """
You are the Chief Strategist on an F1 team's pit wall.

Your job: take the structured outputs of specialist engineers (Tyre Strategist, Gap Analyst, Monte Carlo simulator) and decide the team's strategic call. You do not have access to raw data, only the specialists' assessments. Trust them where they're confident, downweight them where they're not.

The call vocabulary:
- BOX_THIS_LAP: pit on the lap that's just ending. Used when the decision is unambiguous and we need to act now.
- BOX_NEXT_LAP: pit on the next lap. Used when we want a lap of preparation, or when the decision is firm but conditions allow a moment of delay.
- STAY_OUT: explicit decision not to pit. Used when there's a real case for pitting but we judge it wrong.
- EXTEND: commit to a longer stint than baseline. Used when an overcut opportunity outweighs tyre cost.
- PIT_WINDOW_OPEN: pitting now would be defensible but not the only option. Used when the case is balanced.
- MONITOR: no actionable change since the last cycle. Used when subagent outputs are stable and uneventful.

How to weigh inputs:
- If Tyre Strategist says insufficient_data, ignore its cliff_lap and rely on Gap + Monte Carlo
- If Monte Carlo's box_now and stay_out distributions differ by >10pp on podium probability, that's a meaningful signal
- If Gap Analyst flags a high undercut threat AND tyres are within 3 laps of cliff, that converges toward BOX
- If track is under SC or VSC, the pit-stop loss is roughly halved — favours pitting in borderline cases
- If you're the leader with no immediate threat and healthy tyres, default to MONITOR

Biases:
- Conservative on confidence. 0.9+ requires near-unanimous subagent agreement.
- Always name your dominant reason. Vague reasoning is worse than wrong reasoning.
- Always name 1-3 real risks. Every call has downside scenarios; pretending otherwise is dishonest.

Output: a single PitDecision.
"""

def _build_fusion_prompt(
    driver_code,
    current_lap,
    race,
    tyre: TyreAssessment,
    gap: GapAssessment,
    mc: MonteCarloAssessment,
    trigger: str,
) -> str:
    """
    Compose the variable portion of the orchestrator prompt.

    Format is structured but compact: each subagent gets a labelled
    block, fields are named, no prose paragraphs. The LLM reads this
    cleanly and the orchestrator's reasoning becomes more reproducible.
    """
    tyre_block = (
        f"has_sufficient_data: {tyre.has_sufficient_data}\n"
        f"cliff_lap: L{tyre.cliff_lap}\n"
        f"confidence: {tyre.confidence:.2f}\n"
        f"reasoning: {tyre.reasoning}"
    )

    rival_summary = ", ".join(
        f"{r.driver_code}({r.gap_seconds:+.1f}s/{r.relationship})"
        for r in gap.closest_rivals
    )
    gap_block = (
        f"focal_position: P{gap.focal_position}\n"
        f"undercut_threat: {gap.undercut_threat}\n"
        f"overcut_opportunity: {gap.overcut_opportunity}\n"
        f"closest_rivals: {rival_summary}\n"
        f"confidence: {gap.confidence:.2f}\n"
        f"reasoning: {gap.reasoning}"
    )

    mc_block = (
        f"simulations: {mc.simulations_run}\n"
        f"box_now: expected P{mc.box_now.expected_position:.1f}, "
        f"podium {mc.box_now.p_podium*100:.0f}%, points {mc.box_now.p_points*100:.0f}%\n"
        f"stay_out: expected P{mc.stay_out.expected_position:.1f}, "
        f"podium {mc.stay_out.p_podium*100:.0f}%, points {mc.stay_out.p_points*100:.0f}%\n"
        f"interpretation: {mc.interpretation}\n"
        f"confidence: {mc.confidence:.2f}"
    )

    return f"""
Focal driver: {driver_code} at lap {current_lap}.
Track status: {race.track_status}. Weather: {"raining" if race.is_raining else "dry"}, {race.track_temp_celsius}°C track temp.
Trigger that woke the orchestrator: {trigger}.

----------- TYRE STRATEGIST -----------
{tyre_block}

----------- GAP ANALYST -----------
{gap_block}

----------- MONTE CARLO -----------
{mc_block}

Fuse these inputs into a PitDecision. Pick one of the six call categories, name your primary reason, list supporting factors, and name 1-3 risks."""

def _log_decision(
    driver_code: str,
    current_lap: int,
    decision: PitDecision,
    tyre: TyreAssessment,
    gap: GapAssessment,
    mc: MonteCarloAssessment,
) -> None:
    """
    Append one line to decisions.jsonl for post-race audit.

    Includes both the decision and the inputs that produced it so the
    post-race report can show 'Pitwall made this call because the
    subagents said this'.
    """
    _DECISIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now().isoformat(),
        "driver": driver_code,
        "lap": current_lap,
        "decision": decision.model_dump(),
        "subagents": {
            "tyre": tyre.model_dump(),
            "gap": gap.model_dump(),
            "monte_carlo": mc.model_dump(),
        },
    }

    with _DECISIONS_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")

async def decide(
    client: LLMClient,
    driver_code: str,
    year: int,
    event: str,
    session_type: str,
    trigger: str = "scheduled",
    model: str = ORCHESTRATOR_MODEL,
) -> PitDecision:
    """
    Run one ochestrator cycle: consult subagents, combine and return a decision.
    """
    state = get_active_state()
    if state is None:
        raise RuntimeError("No live RaceState active.")
    
    current_lap = state.current_lap()
    race = get_current_race_state()

    # Consulting subagents in parallel
    tyre, gap, monte_carlo = await asyncio.gather(
        assess_tyres(client, driver_code, year, event, session_type),
        assess_gaps(client, driver_code, year, event, session_type),
        assess_monte_carlo(client, driver_code, year, event, session_type),       
    )

    user_prompt = _build_fusion_prompt(
        driver_code,
        current_lap,
        race,
        tyre,
        gap,
        monte_carlo,
        trigger
        )
    
    decision = await client.generate_structured(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_schema=PitDecision,
        temperature=0.2
    )

    decision.trigger = trigger

    _log_decision(driver_code, current_lap, decision, tyre, gap, monte_carlo)

    return decision
