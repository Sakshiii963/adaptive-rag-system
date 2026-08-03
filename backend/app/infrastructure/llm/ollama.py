"""HTTP adapter for local Ollama generation; streaming is deliberately disabled in Milestone 6."""

import httpx


class OllamaGenerationProvider:
    """Calls only a locally running Ollama server with the configured Qwen2.5 model."""

    def __init__(self, base_url: str, model_name: str, timeout_seconds: float) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str, max_output_tokens: int) -> str:
        """Generate one non-streaming completion from Ollama's local `/api/generate` endpoint."""
        response = httpx.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": max_output_tokens},
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        answer = payload.get("response")
        if not isinstance(answer, str):
            raise RuntimeError("Ollama response did not contain a text response.")
        return answer.strip()
