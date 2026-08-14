"""Provider-neutral LLM access for VULCAN.

Every LLM call in the pipeline goes through :class:`LLMClient`, whose adapters
cover both closed-source and open-source models:

===================  ==========================================================
``openai``           GPT models via the OpenAI API
``anthropic``        Claude models via the Anthropic Messages API
``gemini``           Gemini models via the Google Generative AI API
``openai_compatible``  Any OpenAI-compatible server — vLLM, SGLang, Ollama,
                     llama.cpp, TGI, LM Studio — which is how open-weight
                     models (Qwen, Llama, Mistral, DeepSeek, …) are served
===================  ==========================================================

Adapters return one normalised, OpenAI-shaped response so no pipeline step ever
branches on which model produced it. See :mod:`vulcan.llm.base` for that
contract and ``docs/providers.md`` for setup.
"""

from .base import LLMClient
from .capabilities import ModelCapabilities, detect as detect_capabilities
from .factory import (
    PROVIDERS,
    close_clients,
    create_client,
    load_dotenv,
    resolve_llm_config,
)

__all__ = [
    "LLMClient",
    "ModelCapabilities",
    "detect_capabilities",
    "PROVIDERS",
    "create_client",
    "resolve_llm_config",
    "close_clients",
    "load_dotenv",
]
