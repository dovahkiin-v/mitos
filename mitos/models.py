"""Model selection and abstraction for Mitos.

This module acts as the single source of truth for all LLM and embedding model
references. It implements a two-layer abstraction mapping family keys directly
to concrete IDs, preserving model selection specificity (OD2).
"""

from typing import Dict, Mapping, Optional

# Model Family Keys to concrete Model IDs
MODEL_IDS: Dict[str, str] = {
    "FLASH_LITE": "gemini-3.1-flash-lite",
    "FLASH": "gemini-3.5-flash",
    "SONNET": "claude-sonnet-4-6",
    "EMBEDDING": "gemini-embedding-2"
}


MODEL_ALIASES = ["FLASH_LITE", "FLASH", "SONNET"]
EMBEDDING_DIM = 3072  # Dimension size for gemini-embedding-2


def get_model_id(alias: str, env: Optional[Mapping[str, str]] = None) -> str:
    """Gets the concrete model ID for a given model alias.

    Args:
        alias: One of "FLASH_LITE", "FLASH", "SONNET".
        env: The resolved environment of the workspace this call is *for* —
            ``MitosConfig.env``, whose ``MITOS_MODEL_OVERRIDE_*`` names
            ``config.RESOLVED_ENV_KEYS`` derives from :data:`MODEL_IDS`. ``None``
            means no map was supplied, and then no override applies: this leaf
            reads no process environment, so an override reaches it only by being
            passed (an override that lives in a workspace's ``.env`` belongs to
            *that* workspace, and a module-level read would answer for whichever
            directory the process was launched in).

    Returns:
        The string model identifier (e.g. 'gemini-3.1-flash-lite').
    """
    upper_alias = alias.upper()
    if upper_alias not in MODEL_IDS:
        raise ValueError(
            f"Unsupported model alias: {alias}. Must be one of {MODEL_ALIASES}"
        )

    env_override = _override(env, f"MITOS_MODEL_OVERRIDE_{upper_alias}")
    if env_override:
        return env_override

    return MODEL_IDS[upper_alias]


def get_embedding_model_id(env: Optional[Mapping[str, str]] = None) -> str:
    """Gets the model ID for the embedding model.

    Args:
        env: The resolved environment of the workspace this call is for; see
            :func:`get_model_id`. Named explicitly rather than folded into the
            alias function because ``MODEL_ALIASES`` omits ``EMBEDDING``, and
            this is the override costliest to lose: the embedding cache keys on
            content hash alone, so a mis-routed one reads as working while cached
            prior-generation vectors flow into a new-generation collection.

    Returns:
        The string model identifier for embedding.
    """
    env_override = _override(env, "MITOS_MODEL_OVERRIDE_EMBEDDING")
    if env_override:
        return env_override
    return MODEL_IDS["EMBEDDING"]


def _override(env: Optional[Mapping[str, str]], name: str) -> Optional[str]:
    """Reads one override out of a possibly-absent map.

    ``env is None`` rather than ``env or {}``: an empty map and an absent one do
    behave alike here, but the ``or`` spelling is the shape that has already
    produced two defects in this vision, and its callers' truthiness test on the
    *value* below is the one that must stay exact — an empty override must leave
    the baseline id in force, not blank it.
    """
    if env is None:
        return None
    return env.get(name)
