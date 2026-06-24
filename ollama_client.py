"""
Ollama LLM client for streaming and non-streaming chat completions.
Uses standard library urllib — no third-party HTTP dependency required.
"""

import json
import urllib.request
import urllib.error
import logging

import config

logger = logging.getLogger(__name__)


def query_ollama_stream(prompt: str):
    """
    Queries local Ollama endpoint '/api/chat' and yields tokens sequentially.
    BUG FIX: Added configurable timeout (default 120s) to prevent indefinite hangs.
    """
    url = f"{config.OLLAMA_BASE_URL}/api/chat"

    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }

    headers = {"Content-Type": "application/json"}

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        response = urllib.request.urlopen(req, timeout=config.OLLAMA_TIMEOUT_SECONDS)
        for line in response:
            if line:
                decoded_line = line.decode("utf-8").strip()
                if not decoded_line:
                    continue
                try:
                    data = json.loads(decoded_line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    logger.error(
                        f"Could not parse Ollama response line: {decoded_line}"
                    )
                    continue
    except urllib.error.URLError as e:
        logger.error(f"Ollama server connection failed at {url}: {e}")
        yield (
            f"Error: Unable to connect to local Ollama server at "
            f"{config.OLLAMA_BASE_URL}. Ensure Ollama is running and "
            f"{config.OLLAMA_MODEL} is pulled."
        )
    except TimeoutError:
        logger.error(
            f"Ollama request timed out after {config.OLLAMA_TIMEOUT_SECONDS}s"
        )
        yield "Error: Ollama request timed out. The model may be overloaded."
    except Exception as e:
        logger.error(f"Unexpected error during Ollama stream query: {e}")
        yield f"Error: Unexpected LLM client error: {e}"


def query_ollama_non_stream(prompt: str) -> str:
    """
    Queries local Ollama endpoint '/api/chat' and returns the full response string.
    BUG FIX: Added configurable timeout (default 120s) to prevent indefinite hangs.
    """
    url = f"{config.OLLAMA_BASE_URL}/api/chat"

    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    headers = {"Content-Type": "application/json"}

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            req, timeout=config.OLLAMA_TIMEOUT_SECONDS
        ) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data.get("message", {}).get("content", "")
    except urllib.error.URLError as e:
        logger.error(f"Ollama server connection failed at {url}: {e}")
        return (
            f"Error: Unable to connect to local Ollama server at "
            f"{config.OLLAMA_BASE_URL}."
        )
    except TimeoutError:
        logger.error(
            f"Ollama request timed out after {config.OLLAMA_TIMEOUT_SECONDS}s"
        )
        return "Error: Ollama request timed out. The model may be overloaded."
    except Exception as e:
        logger.error(f"Unexpected error during Ollama non-stream query: {e}")
        return f"Error: Unexpected LLM client error: {e}"
