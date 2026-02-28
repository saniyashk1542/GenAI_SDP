"""Simple client for calling a local Ollama model over HTTP."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


OLLAMA_URL = "http://localhost:11434/api/generate"


def generate_text(prompt: str, model: str = "llama3.2", temperature: float = 0.6) -> str:
    """Send a prompt to Ollama and return the generated text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as error:
        raise RuntimeError(
            "Could not connect to Ollama. Make sure Ollama is running locally."
        ) from error

    parsed = json.loads(body)
    text = parsed.get("response", "").strip()
    if not text:
        raise RuntimeError("Ollama returned an empty response.")
    return text
