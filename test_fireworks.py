"""Test Fireworks GLM 5.2 JSON output directly."""
import os
from openai import OpenAI

key = os.environ.get("FIREWORKS_API_KEY")
if not key:
    print("ERROR: FIREWORKS_API_KEY not set")
    exit(1)

client = OpenAI(api_key=key, base_url="https://api.fireworks.ai/inference/v1")
r = client.chat.completions.create(
    model="accounts/fireworks/models/glm-5p2",
    max_tokens=500,
    messages=[
        {"role": "system", "content": "You are a CAD drafting expert. Respond with valid JSON only."},
        {"role": "user", "content": 'Return this JSON: {"dimensions": [{"type": "linear", "from": [0,0], "to": [10,0], "label": "10mm"}], "notes": ["test"]}'},
    ],
    response_format={"type": "json_object"},
)
print(repr(r.choices[0].message.content))
