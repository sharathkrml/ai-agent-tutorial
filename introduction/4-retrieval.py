import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
import requests

# -------------------------------------------------
# 1. Initialize OpenAI client
# -------------------------------------------------
load_dotenv()
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


def search_kb(question: str):
    with open("kb.json", "r") as f:
        return json.load(f)


# -------------------------------------------------
# 2. setup tool
# -------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Search the knowledge base for the answer to the question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to search the knowledge base for.",
                    },
                },
            },
            "strict": True,
        },
    }
]


# -------------------------------------------------
# 3. setup prompt
# -------------------------------------------------

system_prompt = "You are a helpful assistant that answers questions from the knowledge base about our e-commerce store."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the return policy?"},
]

completion = client.chat.completions.create(
    model=modelName,
    messages=messages,
    tools=tools,
)
# --------------------------------------------------------------
# Step 2: Model decides to call function(s)
# --------------------------------------------------------------

completion.model_dump()
# --------------------------------------------------------------
# Step 3: Execute search_kb function
# --------------------------------------------------------------


def call_function(name, args):
    if name == "search_kb":
        return search_kb(**args)


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

completion_2 = client.chat.completions.create(
    model=modelName,
    messages=messages,
)
# --------------------------------------------------------------
# Step 5: Check model response
# --------------------------------------------------------------

print(completion_2.choices[0].message.content)


# --------------------------------------------------------------
# Question that doesn't trigger the tool
# --------------------------------------------------------------

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the weather in Tokyo?"},
]

completion_3 = client.beta.chat.completions.parse(
    model=modelName,
    messages=messages,
    tools=tools,
)

print(completion_3.choices[0].message.content)
