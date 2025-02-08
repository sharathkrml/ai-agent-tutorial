import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

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

# -------------------------------------------------
# Define response schema
# -------------------------------------------------


class CalendarEvent(BaseModel):
    name: str = Field(description="The name of the event.")
    date: str = Field(description="The date of the event.")
    participants: list[str] = Field(description="The participants of the event.")


# -------------------------------------------------
# Define prompt
# -------------------------------------------------

completion_2 = client.beta.chat.completions.parse(
    model=modelName,
    messages=[
        {"role": "system", "content": "Extract the event information."},
        {
            "role": "user",
            "content": "Alice and Bob are going to a science fair on Friday.",
        },
    ],
    response_format=CalendarEvent,
)

print(completion_2.choices[0].message.parsed)
