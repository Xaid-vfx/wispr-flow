class ContextManager:
    """
    Maintains a rolling window of the last N cleaned sentences.
    This is fed to the LLM so it can resolve pronouns, infer topic,
    and maintain consistency across utterances.
    """

    def __init__(self, config):
        self.config = config
        self._sentences: list[str] = []

    def add(self, text: str):
        text = text.strip()
        if text:
            self._sentences.append(text)
            if len(self._sentences) > self.config.max_sentences:
                self._sentences.pop(0)

    def get_context(self) -> str:
        return " ".join(self._sentences)

    def clear(self):
        self._sentences.clear()

    def __len__(self):
        return len(self._sentences)
