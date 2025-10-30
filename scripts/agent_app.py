# agent_app.py
# --------------------------------------------------------------
import os
import json
import openai
from retrieval_backend import snippets, ready


# IMPORTANT: Set SSL environment variables BEFORE importing google.generativeai
# This fixes SSL_ERROR_SSL with self-signed certificates in corporate proxies
import certifi
os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''

# For corporate proxies with SSL inspection, disable certificate verification
os.environ['GRPC_SSL_CIPHER_SUITES'] = 'HIGH'
os.environ['CURL_CA_BUNDLE'] = ''

import ssl
import grpc
import urllib3
import httpx
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Create an unverified SSL context for Google services only
# DO NOT apply globally as it breaks OpenAI API connections
_unverified_ssl_context = ssl._create_unverified_context

# NOW import Google Generative AI after SSL setup

# Load API key from config2.json
with open("config2.json", "r") as f:
    config = json.load(f)

OPENAI_API_KEY = config["OPENAI_API_KEY"]
MODEL = "gpt-4o"          

def ask_llm(prompt: str) -> str:
    # Create httpx client with SSL verification disabled for corporate proxies
    http_client = httpx.Client(
        verify=False,  # Disable SSL verification for corporate proxy
        timeout=60.0
    )
    
    client = openai.OpenAI(
        api_key=OPENAI_API_KEY,
        http_client=http_client
    )
    
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
