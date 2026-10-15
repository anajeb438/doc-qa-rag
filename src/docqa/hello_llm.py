import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # reads your .env file so the key is available

client = Groq(api_key=os.getenv("LLM_API_KEY"))

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "Explain what RAG is in one sentence."}],
)

print(response.choices[0].message.content)