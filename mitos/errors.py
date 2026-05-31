"""Custom exception definitions for Mitos.

This module contains the hierarchical exception architecture used across
the Mitos codebase to ensure precise error vectors and graceful degradation.
"""

class MitosError(Exception):
    """Base exception class for all Mitos errors."""
    pass


class ParseError(MitosError):
    """Raised when the deterministic markdown parser encounters malformed input.

    Attributes:
        line_start: The line number where the offending entry starts.
        line_end: The line number where the offending entry ends.
        message: Detailed explanation of the parsing failure.
    """

    def __init__(self, message: str, line_start: int = 1, line_end: int = 1) -> None:
        self.line_start = line_start
        self.line_end = line_end
        self.message = message
        super().__init__(f"Lines {line_start}-{line_end}: {message}")


class ValidationError(MitosError):
    """Raised when a node or relationship violates architectural invariants."""
    pass


class DatabaseError(MitosError):
    """Raised when SQLite database operations fail or encounter lock contention."""
    pass


class VectorStoreError(MitosError):
    """Raised when Qdrant vector store operations fail."""
    pass


class EmbeddingError(MitosError):
    """Raised when embedding provider API calls fail."""
    pass


class SynthesisError(MitosError):
    """Raised when the LLM synthesis or enrichment call fails."""
    pass
