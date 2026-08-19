"""
This script is used to get the image from the Langfuse media API, given the media ID.
"""

import requests
import os

from knowledge_storm.utils import load_api_key
current_dir = os.path.dirname(os.path.abspath(__file__))
secrets_path = os.path.join(current_dir, '..', 'secrets.toml')
load_api_key(toml_file_path=os.path.abspath(secrets_path))

base_URL = os.getenv("LANGFUSE_HOST")
public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
secret_key = os.getenv("LANGFUSE_SECRET_KEY")

media_id = "Gy7Kl38bWLGm7VzlEIb9AF"

media_request = requests.get(
    f"{base_URL}/api/public/media/{media_id}",
    auth=(public_key or "", secret_key or "")
)
 
media_response = media_request.json()
print(media_response)