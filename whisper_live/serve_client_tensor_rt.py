import json
import threading
import time

from loguru import logger

from whisper_live.serve_client_base import ServeClientBase
from whisper_live.transcriber_tensorrt import WhisperTRTLLM


class ServeClientTensorRT(ServeClientBase):
    SINGLE_MODEL = None
    SINGLE_MODEL_LOCK = threading.Lock()

    def __init__(self, websocket, task="transcribe", multilingual=False, language=None, client_uid=None, model=None,
                 single_model=False):
        """
        Initialize a ServeClient instance.
        The Whisper model is initialized based on the client's language and device availability.
        The transcription thread is started upon initialization. A "SERVER_READY" message is sent
        to the client to indicate that the server is ready.

        Args:
            websocket (WebSocket): The WebSocket connection for the client.
            task (str, optional): The task type, e.g., "transcribe." Defaults to "transcribe".
            device (str, optional): The device type for Whisper, "cuda" or "cpu". Defaults to None.
            multilingual (bool, optional): Whether the client supports multilingual transcription. Defaults to False.
            language (str, optional): The language for transcription. Defaults to None.
            client_uid (str, optional): A unique identifier for the client. Defaults to None.
            single_model (bool, optional): Whether to instantiate a new model for each client connection. Defaults to False.

        """
        super().__init__(client_uid, websocket)
        self.language = language if multilingual else "en"
        self.task = task
        self.eos = False

        if single_model:
            if ServeClientTensorRT.SINGLE_MODEL is not None:
                logger.info("Using preloaded TensorRT model")
                self.transcriber = ServeClientTensorRT.SINGLE_MODEL
            else:
                logger.info("TensorRT single model was requested but not preloaded. Creating model...")
                self.create_model(model, multilingual)
                ServeClientTensorRT.SINGLE_MODEL = self.transcriber
        else:
            self.create_model(model, multilingual)

        # threading
        self.trans_thread = threading.Thread(target=self.speech_to_text)
        self.trans_thread.start()

        self.websocket.send(json.dumps({
            "uid": self.client_uid,
            "message": self.SERVER_READY,
            "backend": "tensorrt"
        }))

    def create_model(self, model, multilingual, warmup=True):
        """
        Instantiates a new model, sets it as the transcriber and does warmup if desired.
        """
        # Ensure language is never None
        language = self.language if self.language is not None else "en"

        self.transcriber = WhisperTRTLLM(
            model,
            assets_dir="assets",
            device="cuda",
            is_multilingual=multilingual,
            language=language,
            task=self.task
        )
        if warmup:
            self.warmup()

    def warmup(self, warmup_steps=10):
        """
        Warmup TensorRT since first few inferences are slow.

        Args:
            warmup_steps (int): Number of steps to warm up the model for.
        """
        logger.info("[INFO:] Warming up TensorRT engine..")
        mel, _ = self.transcriber.log_mel_spectrogram("assets/jfk.flac")
        for i in range(warmup_steps):
            self.transcriber.transcribe(mel)

    def set_eos(self, eos):
        """
        Sets the End of Speech (EOS) flag.

        Args:
            eos (bool): The value to set for the EOS flag.
        """
        self.lock.acquire()
        self.eos = eos
        self.lock.release()

    def handle_transcription_output(self, transcription_result, audio_duration):
        """
        Handle the transcription output, updating the transcript and sending data to the client.

        Args:
            transcription_result (str): The last segment from the whisper output which is considered to be incomplete because
                                of the possibility of word being truncated.
            audio_duration (float): Duration of the transcribed audio chunk.
        """
        segments = self.prepare_segments(
            {"text": transcription_result, "confidence": 0.0})  # TensorRT doesn't provide confidence directly
        self.send_transcription_to_client(segments)
        if self.eos:
            self.update_timestamp_offset(transcription_result, audio_duration)

    def transcribe_audio(self, audio_input):
        """
        Transcribes the provided audio sample using the configured transcriber instance.

        If the language has not been set, it updates the session's language based on the transcription
        information.

        Args:
            audio_input (np.array): The audio chunk to be transcribed. This should be a NumPy
                                    array representing the audio data.

        Returns:
            The transcription result from the transcriber. The exact format of this result
            depends on the implementation of the `transcriber.transcribe` method but typically
            includes the transcribed text.
        """
        if ServeClientTensorRT.SINGLE_MODEL:
            ServeClientTensorRT.SINGLE_MODEL_LOCK.acquire()
        logger.info(f"[WhisperTensorRT:] Processing audio with duration: {audio_input.shape[0] / self.RATE}")
        mel, duration = self.transcriber.log_mel_spectrogram(audio_input)
        last_segment = self.transcriber.transcribe(
            mel,
            text_prefix=f"<|startoftranscript|><|{self.language}|><|{self.task}|><|notimestamps|>"
        )
        if ServeClientTensorRT.SINGLE_MODEL:
            ServeClientTensorRT.SINGLE_MODEL_LOCK.release()
        if last_segment:
            self.handle_transcription_output(last_segment, duration)

    def speech_to_text(self):
        """
        Process an audio stream in an infinite loop, continuously transcribing the speech.

        This method continuously receives audio frames, performs real-time transcription, and sends
        transcribed segments to the client via a WebSocket connection.

        If the client's language is not detected, it waits for 30 seconds of audio input to make a language prediction.
        It utilizes the Whisper ASR model to transcribe the audio, continuously processing and streaming results. Segments
        are sent to the client in real-time, and a history of segments is maintained to provide context.Pauses in speech
        (no output from Whisper) are handled by showing the previous output for a set duration. A blank segment is added if
        there is no speech for a specified duration to indicate a pause.

        Raises:
            Exception: If there is an issue with audio processing or WebSocket communication.

        """
        while True:
            if self.exit:
                logger.info("Exiting speech to text thread")
                break

            if self.frames_np.shape[0] == 0:
                time.sleep(0.02)  # wait for any audio to arrive
                continue

            self.clip_audio_if_no_valid_segment()

            input_bytes, duration = self.get_audio_chunk_for_processing()
            if duration < 0.4:
                continue

            try:
                input_sample = input_bytes.copy()
                logger.info(f"[WhisperTensorRT:] Processing audio with duration: {duration}")
                self.transcribe_audio(input_sample)

            except Exception as e:
                logger.error(f"[ERROR]: {e}")