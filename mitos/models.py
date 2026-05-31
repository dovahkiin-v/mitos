"""Model selection and abstraction for Mitos.

This module acts as the single source of truth for all LLM and embedding model
references. It implements a two-layer abstraction mapping family keys directly
to concrete IDs, preserving model selection specificity (OD2).
"""

import os
from typing import Dict

# Model Family Keys to concrete Model IDs
MODEL_IDS: Dict[str, str] = {
    "FLASH_LITE": "gemini-3.1-flash-lite",
    "FLASH": "gemini-3.5-flash",
    "SONNET": "claude-sonnet-4-6",
    "EMBEDDING": "gemini-embedding-2"
}


CAPABILITY_TIERS = ["FLASH_LITE", "FLASH", "SONNET"]


def get_model_id(alias: str) -> str:
    """Gets the concrete model ID for a given model alias.

    Args:
        alias: One of "FLASH_LITE", "FLASH", "SONNET".

    Returns:
        The string model identifier (e.g. 'gemini-3.1-flash-lite').
    """
    upper_alias = alias.upper()
    if upper_alias not in MODEL_IDS:
        raise ValueError(
            f"Unsupported model alias: {alias}. Must be one of {CAPABILITY_TIERS}"
        )

    env_override = os.environ.get(f"MITOS_MODEL_OVERRIDE_{upper_alias}")
    if env_override:
        return env_override

    return MODEL_IDS[upper_alias]


def get_embedding_model_id() -> str:
    """Gets the model ID for the embedding model.

    Returns:
        The string model identifier for embedding.
    """
    env_override = os.environ.get("MITOS_MODEL_OVERRIDE_EMBEDDING")
    if env_override:
        return env_override
    return MODEL_IDS["EMBEDDING"]
