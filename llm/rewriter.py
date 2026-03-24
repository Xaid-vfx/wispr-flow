import ollama

SYSTEM_PROMPT = """\
You are a personal dictation assistant. Your job is to turn raw speech-to-text \
output into clean, natural written text.

Rules:
- Fix grammar, punctuation, and capitalization
- Remove filler words (um, uh, like, you know) unless they seem intentional
- Preserve the original meaning exactly — do not add, remove, or reinterpret ideas
- Match the tone implied by the context (casual, professional, etc.)
- Return ONLY the cleaned text — no explanations, no quotes, no commentary
- If the input is already clean, return it as-is
- CRITICAL: The context shows previous sentences only to help resolve references \
(e.g. "that", "it", "them"). Do NOT use context to expand, lengthen, or add \
information that is not present in the raw speech input. Your output must be \
derivable entirely from the raw speech input alone. If the input is short, the \
output must also be short."""

USER_TEMPLATE = """\
Context (what was said before — use this to resolve ambiguous references):
{context}

Raw speech input:
{transcript}

Cleaned text:"""


class LLMRewriter:
    def __init__(self, config):
        self.config = config
        self._client: ollama.Client | None = None

    def load(self):
        """Connect to Ollama and warm up the model with a short prompt."""
        try:
            self._client = ollama.Client(host=self.config.base_url)
            self._client.generate(
                model=self.config.model,
                prompt="Hello.",
                options={"num_predict": 5},
            )
        except Exception as e:
            print(f"  Warning: LLM not reachable ({e}). Delta prefix will fall back to raw.", flush=True)
            self._client = None

    def rewrite(self, transcript: str, context: str = "") -> str:
        """
        Rewrite a raw transcript using context from previous sentences.
        Falls back to the raw transcript if LLM is unavailable.
        """
        transcript = transcript.strip()
        if not transcript:
            return transcript
        if not self._client:
            print("  ⚠  LLM not connected — delta prefix ignored. Is Ollama running?", flush=True)
            return transcript

        prompt = USER_TEMPLATE.format(
            context=context if context else "(none — this is the start of the session)",
            transcript=transcript,
        )

        try:
            response = self._client.generate(
                model=self.config.model,
                system=SYSTEM_PROMPT,
                prompt=prompt,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            )
            result = response.response.strip()
            # Guard: empty or single-char response → use raw
            if not result or len(result) < 2:
                return transcript
            # Guard: output >2.5x longer than input → LLM hallucinated from context
            if len(result) > len(transcript) * 2.5 + 40:
                return transcript
            return result
        except Exception as e:
            print(f"  [LLM error: {e}]")
            return transcript
