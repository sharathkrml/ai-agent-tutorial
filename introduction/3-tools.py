import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
import requests

load_dotenv()
# -------------------------------------------------
# 1. Initialize OpenAI client
# -------------------------------------------------
apiKey = os.getenv("API_KEY")
mode = os.getenv("MODE")
if mode == "LOCAL":
    base_url = "http://localhost:11434/v1"
    api_key = "ollama"
    modelName = "deepseek-r1:7b"
else:
    base_url = "https://openrouter.ai/api/v1"
    api_key = apiKey
    modelName = "google/gemini-2.0-flash-exp:free"

client = OpenAI(
    base_url=base_url,
    api_key=apiKey,
    # base_url="http://localhost:11434/v1",
    # api_key="ollama",  # required, but unused
)


"""
docs: https://platform.openai.com/docs/guides/function-calling
"""

# --------------------------------------------------------------
# Define the tool (function) that we want to call
# --------------------------------------------------------------


def get_weather(latitude, longitude):
    """This is a publically available API that returns the weather for a given location."""
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    data = response.json()
    return data["current"]


# --------------------------------------------------------------
# Step 1: Call model with get_weather tool defined
# --------------------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for provided coordinates in celsius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

system_prompt = """"
You are a helpful weather assistant.

Longitude and latitude of few cities are given below.

Paris: 48.8566, 2.3522
London: 51.5074, -0.0901
New York: 40.7128, -74.0060
Tokyo: 35.6812, 139.7671


"""

messages = [
    {"role": "system", "content": system_prompt},
    # {"role": "user", "content": "What's the weather like in Paris today?"},
    {"role": "user", "content": "What's the weather like in Paris today?"},
]

completion = client.chat.completions.create(
    model=modelName,
    messages=messages,
    tools=tools,
)

# --------------------------------------------------------------
# Step 2: Model decides to call function(s)
# --------------------------------------------------------------

# print(completion.model_dump())

# --------------------------------------------------------------
# Step 3: Execute get_weather function
# --------------------------------------------------------------


def call_function(name, args):
    if name == "get_weather":
        return get_weather(**args)


for tool_call in completion.choices[0].message.tool_calls:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    messages.append(completion.choices[0].message)

    result = call_function(name, args)
    messages.append(
        {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
    )


# --------------------------------------------------------------
# Step 4: Supply result and call model again
# --------------------------------------------------------------
completion = client.chat.completions.create(
    model=modelName,
    messages=messages,
)

print(completion.choices[0].message)
