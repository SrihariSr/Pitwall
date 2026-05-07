"""
First Gemini API call
Check that the key, SDK, and connection all work.
"""
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in .env — check the file exists and the name is exact")

client = genai.Client(api_key=api_key)

print("Calling Gemini...")
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="In one sentence, what is the most famous strategy mistake in F1 history?",
)

print("\nGemini's response:")
print(response.text)