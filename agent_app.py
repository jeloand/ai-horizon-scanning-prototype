# agent_app.py
# --------------------------------------------------------------
import os
import json
import openai
from retrieval_backend import snippets, ready

# Load API key from config2.json
with open("config2.json", "r") as f:
    config = json.load(f)

OPENAI_API_KEY = config["OPENAI_API_KEY"]
MODEL = "gpt-4o"          

def ask_llm(prompt: str) -> str:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model       = MODEL,
        messages    = [{"role": "user", "content": prompt}],
        temperature = 0.3,
        max_tokens  = 512,
    )
    return resp.choices[0].message.content.strip()

def main():
    if not ready():
        print("❌ Retrieval index not found – run the scraper first.")
        return

    while True:
        user_q = input("\n🔍  Ask a labour-policy question (or 'quit'): ").strip()
        if user_q.lower() in {"q", "quit", "exit"}:
            break

        context = snippets(user_q, k=5)
        prompt  = f"""You are a concise labour-market policy assistant.

### Context
{context}

### Question
{user_q}

### Answer (use bullet points when helpful, be concise):
"""
        print("\n💬  GPT-4-o:\n", ask_llm(prompt))

if __name__ == "__main__":
    main()
