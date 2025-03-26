from dataclasses import dataclass
from typing import Optional, List, Tuple

from faster_whisper.vad import VadOptions

from whisper_live.transcription_options import TranscriptionOptions


@dataclass
class TranscriptionInfo:
    language: str
    language_probability: float
    duration: float
    duration_after_vad: float
    all_language_probs: Optional[List[Tuple[str, float]]]
    transcription_options: TranscriptionOptions
    vad_options: VadOptions
