# embed_loader.py
# --------------------------------------------------
# Load the SentenceTransformer only once (lazy singleton)

import os
import ssl
from functools import lru_cache
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Disable SSL verification globally for requests library
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''

# Monkey-patch requests to disable SSL verification
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    import urllib3
    
    # Disable SSL warnings
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Store original methods
    _original_request = requests.Session.request
    
    def patched_request(self, method, url, **kwargs):
        """Patched request method that disables SSL verification"""
        kwargs['verify'] = False
        return _original_request(self, method, url, **kwargs)
    
    # Apply the patch
    requests.Session.request = patched_request
except Exception:
    pass

# Also disable SSL verification at the SSL module level
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

@lru_cache(maxsize=1)
def get_embedder():
    """Returns the singleton SentenceTransformer model."""
    return SentenceTransformer(EMBED_MODEL_NAME)
