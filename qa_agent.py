import importlib
import inspect
import os
from abc import ABC, abstractmethod


class BaseQAAgent(ABC):
    @abstractmethod
    def answer_question(self, message):
        raise NotImplementedError

    @abstractmethod
    def direct_ask(self, question, timestamp=None):
        raise NotImplementedError

    def process_items(self):
        return None

    def refresh_from_state(self):
        return None

    def set_game_name(self, game_name):
        return None


def create_qa_agent(state, disabled=False):
    """Create the configured QA agent.

    QA_AGENT supports:
      - "gemini" for the bundled Gemini implementation
      - "none", "off", or "disabled" to disable QA
      - "module:ClassName" for a custom API or local-model adapter
    """
    provider = (state.config.qa_agent or "gemini").strip()
    provider_key = provider.lower()

    if disabled or provider_key in ("none", "off", "disabled"):
        return None

    if provider_key == "gemini":
        api_key = os.getenv("GENAI_API_KEY")
        if not api_key:
            raise ValueError("GENAI_API_KEY is required when QA_AGENT = 'gemini'.")
        from gemini_agent import GeminiAgent

        return GeminiAgent(api_key=api_key, state=state)

    agent_cls = _load_agent_class(provider)
    return _instantiate_agent(agent_cls, state)


def _load_agent_class(target):
    if ":" not in target:
        raise ValueError(
            "Custom QA_AGENT values must use 'module:ClassName', "
            "for example 'local_agent:LocalAgent'."
        )

    module_name, class_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _instantiate_agent(agent_cls, state):
    signature = inspect.signature(agent_cls)
    params = signature.parameters

    if "state" in params or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return agent_cls(state=state)

    if len(params) == 1 or any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in params.values()):
        return agent_cls(state)

    return agent_cls()
