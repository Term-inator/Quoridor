"""Languages supported by user-facing Quoridor interfaces."""

from enum import Enum


class Language(str, Enum):
    """A supported user-interface language."""

    CHINESE = "zh"
    ENGLISH = "en"
