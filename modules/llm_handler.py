"""
modules/llm_handler.py — Module 6: LLM Integration (Ollama)
────────────────────────────────────────────────────────────
Responsibilities:
  - Communicate with a locally hosted Ollama server
  - Send prompts and receive generated completions
  - Handle connection errors and timeouts gracefully
  - Support streaming responses (optional)

Requirements:
  - Ollama must be running: `ollama serve`
  - Desired model must be pulled: `ollama pull mistral`
  - No external API keys required

Ollama REST API endpoint: POST /api/generate
"""

import json
import logging
import requests
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


class LLMHandler:
    """
    Sends prompts to a locally hosted Ollama LLM and returns completions.

    Usage:
        llm = LLMHandler(base_url="http://localhost:11434", model="mistral")
        answer = llm.generate(prompt)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "mistral",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: int = 600,
    ):
        """
        Args:
            base_url: URL of the running Ollama server.
            model: Model name (must be pulled via `ollama pull <model>`).
            temperature: Sampling temperature. Low = more factual/deterministic.
            max_tokens: Maximum tokens in the generated response.
            timeout: HTTP request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        logger.info(
            f"LLMHandler configured — model='{model}', "
            f"base_url='{base_url}', temperature={temperature}"
        )

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def generate(self, prompt: str) -> str:
        """
        Generate a completion for the given prompt (blocking).

        Args:
            prompt: Full formatted prompt string.

        Returns:
            Generated text response from the LLM.

        Raises:
            ConnectionError: If Ollama server is unreachable.
            RuntimeError: On API-level errors.
        """
        endpoint = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        try:
            logger.info(f"Sending prompt to Ollama ({self.model})...")
            response = requests.post(
                endpoint,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            answer = data.get("response", "").strip()
            if not answer:
                logger.warning("Ollama returned an empty response.")
                return "I was unable to generate a response. Please try again."

            logger.info(f"LLM response received ({len(answer)} chars).")
            return answer

        except requests.exceptions.ConnectionError:
            msg = (
                f"Cannot connect to Ollama at {self.base_url}. "
                "Please ensure Ollama is running (`ollama serve`) and the model "
                f"'{self.model}' is pulled (`ollama pull {self.model}`)."
            )
            logger.error(msg)
            raise ConnectionError(msg)

        except requests.exceptions.Timeout:
            msg = f"Ollama request timed out after {self.timeout}s."
            logger.error(msg)
            raise TimeoutError(msg)

        except requests.exceptions.HTTPError as e:
            msg = f"Ollama HTTP error: {e.response.status_code} — {e.response.text}"
            logger.error(msg)
            raise RuntimeError(msg)

        except (json.JSONDecodeError, KeyError) as e:
            msg = f"Failed to parse Ollama response: {e}"
            logger.error(msg)
            raise RuntimeError(msg)

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """
        Generate a streaming completion (yields text tokens as they arrive).

        Args:
            prompt: Full formatted prompt string.

        Yields:
            str — incremental text tokens.
        """
        endpoint = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        try:
            with requests.post(
                endpoint,
                json=payload,
                timeout=self.timeout,
                stream=True,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            token = data.get("response", "")
                            if token:
                                yield token
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue

        except requests.exceptions.ConnectionError:
            yield f"\n\n[ERROR: Cannot reach Ollama at {self.base_url}]"
        except requests.exceptions.Timeout:
            yield "\n\n[ERROR: Request timed out]"

    # ──────────────────────────────────────────
    # Health check
    # ──────────────────────────────────────────

    def is_available(self) -> bool:
        """
        Check whether the Ollama server is reachable and the model is loaded.

        Returns:
            True if the server responds, False otherwise.
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags", timeout=5
            )
            if response.status_code == 200:
                models = [
                    m.get("name", "").split(":")[0]
                    for m in response.json().get("models", [])
                ]
                if self.model in models:
                    logger.info(f"Ollama OK — model '{self.model}' is available.")
                    return True
                else:
                    logger.warning(
                        f"Ollama running but model '{self.model}' not found. "
                        f"Available: {models}. Run: ollama pull {self.model}"
                    )
                    return False
            return False
        except Exception:
            return False

    def list_models(self) -> list:
        """Return list of locally available Ollama model names."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                return [m.get("name") for m in response.json().get("models", [])]
        except Exception:
            pass
        return []
