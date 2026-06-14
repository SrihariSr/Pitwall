"""
Smoke test for the LLM client. Asks Gemini a simple structured question
and validates we get back a properly-typed Pydantic object.
"""
import asyncio
from pydantic import BaseModel, Field

from llm.client import LLMClient


class TyreAssessmentExample(BaseModel):
    """Toy schema to validate the client end-to-end."""
    cliff_lap_estimate: int = Field(description="Lap by which the tyre will hit its performance cliff")
    confidence: float = Field(description="0-1 confidence in the estimate")
    reasoning: str = Field(description="One-sentence justification")


async def main():
    client = LLMClient()

    system = """You are an F1 tyre strategist. Given a stint summary, project
    the lap by which the tyre will hit its performance cliff. Return a single
    JSON object matching the schema. Be conservative — when in doubt, err
    earlier rather than later."""

    user = """Stint: HARD compound, 12 laps old, last 3 lap times 84.1s, 84.3s, 84.7s.
    Track: Monaco, dry, ambient 23C. Project the cliff lap."""

    print("Calling Gemini for structured output...")
    result = await client.generate_structured(
        model="gemini-2.5-flash",
        system_prompt=system,
        user_prompt=user,
        response_schema=TyreAssessmentExample,
    )

    print("\nParsed response:")
    print(f"cliff_lap_estimate: {result.cliff_lap_estimate}")
    print(f"confidence:         {result.confidence}")
    print(f"reasoning:          {result.reasoning}")

    # Demonstrate that the return value is properly typed.
    assert isinstance(result, TyreAssessmentExample)
    assert isinstance(result.cliff_lap_estimate, int)
    print("\nType assertions passed ✓")


if __name__ == "__main__":
    asyncio.run(main())