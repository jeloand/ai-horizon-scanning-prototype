# agent_app.py
# --------------------------------------------------------------
import os
import json
import openai
from pathlib import Path
from dotenv import load_dotenv
from retrieval_backend import snippets, ready

# Load environment variables from .env file
load_dotenv()

# Load API key from environment variable with fallback to config file
def _load_openai_key():
    """Load OpenAI API key from environment or config file."""
    # First try environment variable (for production/App Runner)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    
    # Fallback to config2.json for local development
    config_path = Path(__file__).parent.parent / "config2.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
            return config.get("OPENAI_API_KEY")
    
    raise ValueError(
        "OPENAI_API_KEY not found. Set as environment variable or add to config2.json"
    )

OPENAI_API_KEY = _load_openai_key()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Check if running in production (App Runner) vs local development
IS_PRODUCTION = os.getenv("AWS_EXECUTION_ENV") is not None or os.getenv("DISABLE_SSL_VERIFICATION") != "true"

if not IS_PRODUCTION:
    # ONLY for local development with corporate proxies
    import certifi
    os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = certifi.where()
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    os.environ['GRPC_VERBOSITY'] = 'ERROR'
    os.environ['GRPC_TRACE'] = ''
    os.environ['GRPC_SSL_CIPHER_SUITES'] = 'HIGH'
    os.environ['CURL_CA_BUNDLE'] = ''
    
    import ssl
    import grpc
    import urllib3
    import httpx
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    _unverified_ssl_context = ssl._create_unverified_context

def ask_llm(prompt: str) -> str:
    # Only disable SSL verification for local development with corporate proxies
    if IS_PRODUCTION:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
    else:
        import httpx
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
