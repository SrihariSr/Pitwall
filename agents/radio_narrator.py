"""
The Radio Narrator.

Takes a PitDecision and converts it into a race-engineer radio message;
the kind broadcast over team radio to the driver. This is a post-processor:
runs sequentially after the Orchestrator decides, rather than in parallel
with the other subagents. The Orchestrator decides; the Narrator tells
the driver.
"""

from requests_cache.models import response
import os
from llm.client import LLMClient
from agents.schemas import PitDecision, RadioMessage

DEFAULT_MODEL = os.getenv("LLM_MODEL_SUBAGENT", "gemini-2.5-flash-lite")

SYSTEM_PROMPT = """
You are Pitwall's Radio Narrator. You take a Chief Strategist's pit decision and convert it into the radio message a race engineer would deliver to the driver over team radio.

Your voice:
- Calm. Race engineers do not shout, even on critical calls.
- Concise. The driver is busy.
- Repetitive for critical imperatives. "Box box box" — not just "Box."
- Informative. Always include the WHY in a few words.

You DO NOT:
- Add commentary about reasoning process or strategy theory.
- Use jargon or acronyms the driver does not already know.
- Be theatrical or dramatic.
- Use exclamation marks.
- Mention "the Orchestrator", "the Strategist", or any subagent. You are the team's voice to the driver.

Voice patterns by call:
- BOX_THIS_LAP: "Box this lap, box this lap. [reason]"
- BOX_NEXT_LAP: "Box next lap. [reason]"
- STAY_OUT: "Stay out, stay out. [reason]"
- EXTEND: "We extend. [reason — usually about waiting for opportunity]"
- PIT_WINDOW_OPEN: "Window opens. [conditional or what to watch for]"
- MONITOR: "[Status note]. Monitoring."

Always include the WHY in 5-12 words. Cite specific evidence the decision named: driver codes (e.g. SAI, PER), lap numbers, conditions (yellow flag, rain, drying line) - not generic statements.

Urgency mapping:
- "critical": BOX_THIS_LAP, BOX_NEXT_LAP, immediate-action STAY_OUT
- "info": EXTEND, MONITOR, status-quo updates
- "planning": PIT_WINDOW_OPEN, situational STAY_OUT, heads-up about upcoming events

Output: a RadioMessage with urgency, primary_call (the headline imperative), and full_message (the complete delivered message as a single string).
"""

def _build_user_prompt(decision: PitDecision) -> str:
    supporting = "\n".join(f"- {s}" for s in (decision.supporting_factors or []))
    risks = "\n".join(f"- {r}" for r in (decision.risks or []))

    return f"""
Chief Strategist's decision:

CALL: {decision.call}
CONFIDENCE: {decision.confidence:.2f}
TRIGGER: {decision.trigger}

PRIMARY REASON:
{decision.primary_reason}

SUPPORTING:
{supporting}

RISKS:
{risks}

Generate the race-engineer radio message that delivers this decision to the driver."""

async def narrate(
    client: LLMClient,
    decision: PitDecision,
    model: str = DEFAULT_MODEL
) -> RadioMessage:
    """
    Generate a race-engineer radio message for a PitDecision.

    Higher temperature than the analytical subagents (0.5): natural
    language benefits from a little variety, and we're not asking for
    consistent numerical estimates.
    """

    user_prompt = _build_user_prompt(decision)
    
    return await client.generate_structured(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_schema=RadioMessage,
        temperature=0.5
    )
