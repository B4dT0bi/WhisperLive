from dataclasses import asdict, dataclass
from typing import List, Optional
from warnings import warn

from whisper_live.word import Word


@dataclass
class Segment:
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: Optional[List[Word]]
    temperature: Optional[float]

    def _asdict(self):
        warn(
            "Segment._asdict() method is deprecated, use dataclasses.asdict(Segment) instead",
            DeprecationWarning,
            2,
        )
        return asdict(self)
