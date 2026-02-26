import dspy
import asyncio
import apple_fm_sdk as fm


class AppleFMLM(dspy.LM):
    """
    A custom DSPy Language Model wrapper that routes inference through the
    local, zero-latency apple_fm_sdk.
    """

    def __init__(self, model_name="system_foundation_model"):
        # We pass model to the parent class to satisfy DSPy requirements
        super().__init__(model=model_name)
        self.fm_model = fm.SystemLanguageModel()
        self.kwargs = {
            "temperature": 0.0,  # Not configurable directly in FM SDK but good to declare
            "max_tokens": 1024,
        }
        self.provider = "apple_fm"

    def basic_request(self, prompt: str, **kwargs):
        """Standard request wrapper required by some DSPy flows."""
        session = fm.LanguageModelSession(model=self.fm_model)

        # We must bridge sync and async because DSPy calls this synchronously
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            import nest_asyncio

            nest_asyncio.apply()

        response = asyncio.run(session.respond(prompt))

        # Emulate a DSPy-compatible payload
        return [str(response)]

    def __call__(self, prompt=None, messages=None, **kwargs):
        """The primary execution method for DSPy modules."""
        # DSPy v2.5+ passes 'messages' which is a list of dicts.
        if messages is not None:
            # Flatten the messages into a single prompt string for Apple FM SDK
            prompt_str = "\\n".join([msg.get("content", "") for msg in messages])
        elif prompt is not None:
            prompt_str = prompt
        else:
            raise ValueError("Either prompt or messages must be provided")

        response = self.basic_request(prompt_str, **kwargs)
        self.history.append({"prompt": prompt_str, "response": response})
        return response
