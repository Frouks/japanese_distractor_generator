from enum import Enum, auto


class SentenceContextEnum(Enum):
    """Defines the context types for distractor generation."""
    OPEN = auto()
    CLOSED = auto()
