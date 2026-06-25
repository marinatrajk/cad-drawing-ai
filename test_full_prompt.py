"""Test the full AI dimensioning prompt with Fireworks GLM 5.2."""
import os
import sys
sys.path.insert(0, ".")

from openai import OpenAI
from src.step_parser import load_step, extract_metadata
from src.ai_dim import SYSTEM_PROMPT, _build_user_message

key = os.environ.get("FIREWORKS_API_KEY")
if not key:
    print("ERROR: FIREWORKS_API_KEY not set")
    exit(1)

# Load the bracket part
shape = load_step("samples/bracket.step")
metadata = extract_metadata("samples/bracket.step", shape)
user_msg = _build_user_message(metadata)

print("=== USER MESSAGE ===")
print(user_msg)
print()

client = OpenAI(api_key=key, base_url="https://api.fireworks.ai/inference/v1")

print("=== CALLING GLM 5.2 ===")
r = client.chat.completions.create(
    model="accounts/fireworks/models/glm-5p2",
    max_tokens=2000,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ],
    response_format={"type": "json_object"},
)

content = r.choices[0].message.content
print("=== RAW RESPONSE ===")
print(repr(content))
print()
print("=== FORMATTED ===")
print(content)
