from dataclasses import dataclass
from typing import Optional, Union, List, Iterable


@dataclass
class TranscriptionOptions:
    beam_size: int
    best_of: int
    patience: float
    length_penalty: float
    repetition_penalty: float
    no_repeat_ngram_size: int
    log_prob_threshold: Optional[float]
    no_speech_threshold: Optional[float]
    compression_ratio_threshold: Optional[float]
    condition_on_previous_text: bool
    prompt_reset_on_temperature: float
    temperatures: List[float]
    initial_prompt: Optional[Union[str, Iterable[int]]]
    prefix: Optional[str]
    suppress_blank: bool
    suppress_tokens: Optional[List[int]]
    without_timestamps: bool
    max_initial_timestamp: float
    word_timestamps: bool
    prepend_punctuations: str
    append_punctuations: str
    multilingual: bool
    max_new_tokens: Optional[int]
    clip_timestamps: Union[str, List[float]]
    hallucination_silence_threshold: Optional[float]
    hotwords: Optional[str]