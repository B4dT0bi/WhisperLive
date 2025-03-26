from dataclasses import asdict, dataclass
from warnings import warn

@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float

    def _asdict(self):
        warn(
            "Word._asdict() method is deprecated, use dataclasses.asdict(Word) instead",
            DeprecationWarning,
            2,
        )
        return asdict(self)
