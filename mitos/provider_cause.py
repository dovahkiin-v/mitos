"""One closed classifier for an embedding-provider failure.

When semantic recall degrades, three places say why: the read header
(``lexical.degraded_reason_from_error``), ``surface``'s scope-dump note
(``recall.assess_surface_recall``, which receives the phrase as a string) and the
write-path warning (``sync.py``). They say it from this module and nowhere else,
so for one exception they cannot tell three different stories.

The vocabulary is closed on purpose — ``auth`` / ``quota`` / ``transient`` /
``unknown`` — and is not a plugin point. A cause the rules below cannot see is
``unknown``, whose phrase is the shipped "embedding provider error".

Classification reads typed fields walked down the exception chain, never the
error text, with one exception: the shipped 429 arm, which matches the text and
still wins first. The provider's own text is never returned, because it is a JSON
body that can carry the key's reason codes and project details.

A Tier-1 leaf: stdlib plus ``mitos.errors``. It never imports ``google.*`` or
``httpx`` (``embeddings.py`` loads the SDK lazily, and this module is imported on
the write path); the SDK's exceptions are read by duck typing and by class name.
"""

from typing import Any, Iterator, List, Optional, Tuple

from mitos.errors import EmbeddingError

AUTH: str = "auth"
QUOTA: str = "quota"
TRANSIENT: str = "transient"
UNKNOWN: str = "unknown"

#: The closed vocabulary, in classification order. A row pins it.
PROVIDER_CAUSES: Tuple[str, ...] = (AUTH, QUOTA, TRANSIENT, UNKNOWN)

# One phrase per class. The quota and unknown phrases are byte-identical to the
# strings the read header shipped before this classifier existed. The auth phrase
# never says "down": the provider answered, and it refused.
_PHRASES = {
    AUTH: "embedding provider refused the credentials (auth)",
    QUOTA: "embedding provider rate-limited (429)",
    TRANSIENT: "embedding provider temporarily unavailable (transient)",
    UNKNOWN: "embedding provider error",
}

# ``google.rpc.ErrorInfo`` reasons that mean the credentials or the account were
# refused, sourced from ``googleapis/google/api/error_reason.proto``. An invalid key
# arrives as HTTP 400 ``INVALID_ARGUMENT``, so this field — not the status code —
# is what catches the most common case. An expired key also arrives as
# ``API_KEY_INVALID``; there is no ``API_KEY_EXPIRED`` reason in the enum.
_AUTH_REASONS: Tuple[str, ...] = (
    "API_KEY_INVALID",
    "API_KEY_SERVICE_BLOCKED",
    "API_KEY_HTTP_REFERRER_BLOCKED",
    "API_KEY_IP_ADDRESS_BLOCKED",
    "API_KEY_ANDROID_APP_BLOCKED",
    "API_KEY_IOS_APP_BLOCKED",
    "CONSUMER_SUSPENDED",
    "CONSUMER_INVALID",
    "SERVICE_DISABLED",
    "BILLING_DISABLED",
    "ACCESS_TOKEN_EXPIRED",
    "CREDENTIALS_MISSING",
    "IAM_PERMISSION_DENIED",
)
_AUTH_CODES: Tuple[int, ...] = (401, 403)
_AUTH_STATUSES: Tuple[str, ...] = ("UNAUTHENTICATED", "PERMISSION_DENIED")

_QUOTA_CODE: int = 429
_QUOTA_STATUS: str = "RESOURCE_EXHAUSTED"

_TRANSIENT_STATUSES: Tuple[str, ...] = ("UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL")
# httpx's transport and timeout bases, matched by name so mitos never imports httpx.
# ``TransportError`` covers connect, read, write and protocol failures;
# ``TimeoutException`` covers the timeouts (some of which sit outside it).
_TRANSIENT_CLASS_NAMES: Tuple[str, ...] = ("TimeoutException", "TransportError")

# How many links of ``__cause__``/``__context__`` the walk visits. The production
# chain is two links deep (EmbeddingError from the SDK error); the bound only stops
# a pathological chain from costing anything.
_MAX_CHAIN_DEPTH: int = 8


def _chain(exc: BaseException) -> Iterator[BaseException]:
    """Yields ``exc`` and its causes, preferring ``__cause__`` over ``__context__``."""
    seen = set()
    link: Optional[BaseException] = exc
    while link is not None and id(link) not in seen and len(seen) < _MAX_CHAIN_DEPTH:
        seen.add(id(link))
        yield link
        link = link.__cause__ if link.__cause__ is not None else link.__context__


def _code(link: BaseException) -> Optional[int]:
    code = getattr(link, "code", None)
    return code if isinstance(code, int) and not isinstance(code, bool) else None


def _status(link: BaseException) -> Optional[str]:
    status = getattr(link, "status", None)
    return status if isinstance(status, str) else None


def _reasons(link: BaseException) -> List[str]:
    """Reads ``ErrorInfo`` reasons off an SDK error's ``details``, whatever its shape.

    ``google.genai.errors.APIError.details`` is the response body: ``{"error":
    {..., "details": [...]}}`` on the live REST path, the inner dict with no
    ``"error"`` wrapper on the replay path, a ``{message, status}`` dict for a
    non-JSON body, or a bare string. Every hop is type-checked.
    """
    details: Any = getattr(link, "details", None)
    bodies = details if isinstance(details, list) else [details]
    reasons: List[str] = []
    for body in bodies:
        if not isinstance(body, dict):
            continue
        inner = body.get("error")
        if isinstance(inner, dict):
            body = inner
        items = body.get("details")
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("reason"), str):
                reasons.append(item["reason"])
    return reasons


def _is_quota(link: BaseException) -> bool:
    return _code(link) == _QUOTA_CODE or _status(link) == _QUOTA_STATUS


def _is_auth(link: BaseException) -> bool:
    if _code(link) in _AUTH_CODES or _status(link) in _AUTH_STATUSES:
        return True
    return any(reason in _AUTH_REASONS for reason in _reasons(link))


def _is_transient(link: BaseException) -> bool:
    code = _code(link)
    if code is not None and code >= 500:
        return True
    if _status(link) in _TRANSIENT_STATUSES:
        return True
    if isinstance(link, (TimeoutError, ConnectionError)):
        return True
    return any(cls.__name__ in _TRANSIENT_CLASS_NAMES for cls in type(link).__mro__)


def classify_provider_error(exc: BaseException) -> str:
    """Classifies an embedding-provider failure into one closed cause.

    First match wins, and the order is the contract: quota, then auth, then
    transient, else unknown. Quota is the shipped arm — ``"RESOURCE_EXHAUSTED"``
    or ``"429"`` in ``str(exc)`` — kept first so a wrapped 429 always reads as a
    rate limit; a typed 429 on any chain link lands there too. Auth and transient
    read only typed fields, so a slug or message that happens to contain "403"
    never becomes auth.

    Args:
        exc: The exception that broke the embedding call, usually an
            ``EmbeddingError`` chained from the SDK's error.

    Returns:
        One of :data:`PROVIDER_CAUSES`.
    """
    text = str(exc)
    links = list(_chain(exc))
    if "RESOURCE_EXHAUSTED" in text or "429" in text or any(map(_is_quota, links)):
        return QUOTA
    if any(map(_is_auth, links)):
        return AUTH
    if any(map(_is_transient, links)):
        return TRANSIENT
    return UNKNOWN


def provider_cause_phrase(cause: str) -> str:
    """Returns the one human phrase for a cause class.

    Args:
        cause: One of :data:`PROVIDER_CAUSES`.

    Raises:
        KeyError: ``cause`` is not in the closed vocabulary — a call-site defect.
    """
    return _PHRASES[cause]


def describe_embed_failure(exc: BaseException) -> str:
    """Words a write-path embedding failure without printing the provider's body.

    An ``EmbeddingError`` gets its class phrase. Anything else — a store fault, a
    vector-store error — keeps ``str(exc)`` unchanged, because it is not a provider
    body and its text is the most useful thing to print.

    Args:
        exc: The exception an embedding step raised.
    """
    if isinstance(exc, EmbeddingError):
        return provider_cause_phrase(classify_provider_error(exc))
    return str(exc)
