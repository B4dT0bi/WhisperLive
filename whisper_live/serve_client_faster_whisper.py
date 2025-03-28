import json
import math
import threading
import time

import torch
from loguru import logger

from whisper_live.serve_client_base import ServeClientBase
from whisper_live.whisper_model import WhisperModel


class ServeClientFasterWhisper(ServeClientBase):
    SINGLE_MODEL = None
    SINGLE_MODEL_LOCK = threading.Lock()

    def __init__(self, websocket, task="transcribe", device=None, language=None, client_uid=None,
                 model="large-v3-turbo",
                 initial_prompt=None, vad_parameters=None, use_vad=True, single_model=False):
        """
        Initialize a ServeClient instance.
        The Whisper model is initialized based on the client's language and device availability.
        The transcription thread is started upon initialization. A "SERVER_READY" message is sent
        to the client to indicate that the server is ready.

        Args:
            websocket (WebSocket): The WebSocket connection for the client.
            task (str, optional): The task type, e.g., "transcribe." Defaults to "transcribe".
            device (str, optional): The device type for Whisper, "cuda" or "cpu". Defaults to None.
            language (str, optional): The language for transcription. Defaults to None.
            client_uid (str, optional): A unique identifier for the client. Defaults to None.
            model (str, optional): The whisper model size. Defaults to 'small.en'
            initial_prompt (str, optional): Prompt for whisper inference. Defaults to None.
            single_model (bool, optional): Whether to instantiate a new model for each client connection. Defaults to False.
        """
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
        # Set compute type based on device capabilities
        if self.device == "cuda":
            major, _ = torch.cuda.get_device_capability(self.device)
            self.compute_type = "float16" if major >= 7 else "float32"
        else:
            self.compute_type = "int8"

        ServeClientBase.__init__(self, client_uid, websocket)
        self.language = language
        self.task = task
        self.model_size = model
        self.same_output_threshold = 3
        self.same_output_count = 0
        self.end_time_for_same_output = None
        self.lock = threading.Lock()
        self.transcript = []
        self.text = []
        self.prev_out = ''
        self.current_out = ''
        self.current_segment_avg_logprob = None  # Store the latest segment's avg_logprob
        self.use_vad = use_vad
        self.model_sizes = [
            "tiny", "tiny.en", "base", "base.en", "small", "small.en",
            "medium", "medium.en", "large-v2", "large-v3", "distil-small.en",
            "distil-medium.en", "distil-large-v2", "distil-large-v3",
            "large-v3-turbo", "turbo"
        ]

        self.model_size_or_path = model
        self.language = "en" if self.model_size_or_path.endswith("en") else language
        self.initial_prompt = initial_prompt
        self.vad_parameters = vad_parameters or {"onset": 0.5}
        self.no_speech_thresh = 0.3
        self.same_output_threshold = 10
        self.end_time_for_same_output = None

        # Add this to track the last sent segment
        self.last_sent_segment = None

        if self.model_size_or_path is None:
            return
        logger.info(f"Using Device={self.device} with precision {self.compute_type}")

        try:
            # Check if we should use the preloaded model
            if single_model:
                if ServeClientFasterWhisper.SINGLE_MODEL is not None:
                    logger.info("Using preloaded single model")
                    self.transcriber = ServeClientFasterWhisper.SINGLE_MODEL
                else:
                    logger.info("Single model was requested but not preloaded. Creating model...")
                    self.create_model(self.device)
                    ServeClientFasterWhisper.SINGLE_MODEL = self.transcriber
            else:
                self.create_model(self.device)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.websocket.send(json.dumps({
                "uid": self.client_uid,
                "status": "ERROR",
                "message": f"Failed to load model: {str(self.model_size_or_path)}"
            }))
            self.websocket.close()
            return

        self.use_vad = use_vad

        # threading
        self.trans_thread = threading.Thread(target=self.speech_to_text)
        self.trans_thread.start()
        self.websocket.send(
            json.dumps(
                {
                    "uid": self.client_uid,
                    "message": self.SERVER_READY,
                    "backend": "faster_whisper"
                }
            )
        )

    def create_model(self, device):
        """
        Instantiates a new model, sets it as the transcriber.
        """
        self.transcriber = WhisperModel(
            self.model_size_or_path,
            device=device,
            compute_type=self.compute_type,
            local_files_only=False,
        )

    def check_valid_model(self, model_size):
        """
        Check if it's a valid whisper model size.

        Args:
            model_size (str): The name of the model size to check.

        Returns:
            str: The model size if valid, None otherwise.
        """
        if model_size not in self.model_sizes:
            self.websocket.send(
                json.dumps(
                    {
                        "uid": self.client_uid,
                        "status": "ERROR",
                        "message": f"Invalid model size {model_size}. Available choices: {self.model_sizes}"
                    }
                )
            )
            return None
        return model_size

    def set_language(self, info):
        """
        Updates the language attribute based on the detected language information.

        Args:
            info (object): An object containing the detected language and its probability. This object
                        must have at least two attributes: `language`, a string indicating the detected
                        language, and `language_probability`, a float representing the confidence level
                        of the language detection.
        """
        if info.language_probability > 0.5:
            self.language = info.language
            logger.info(f"Detected language {self.language} with probability {info.language_probability}")
            self.websocket.send(json.dumps(
                {"uid": self.client_uid, "language": self.language, "language_prob": info.language_probability}))

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
        if ServeClientFasterWhisper.SINGLE_MODEL:
            ServeClientFasterWhisper.SINGLE_MODEL_LOCK.acquire()
        result, info = self.transcriber.transcribe(
            audio_input,
            initial_prompt=self.initial_prompt,
            language=self.language,
            task=self.task,
            vad_filter=self.use_vad,
            vad_parameters=self.vad_parameters if self.use_vad else None)
        if ServeClientFasterWhisper.SINGLE_MODEL:
            ServeClientFasterWhisper.SINGLE_MODEL_LOCK.release()

        if self.language is None and info is not None:
            self.set_language(info)
        return result

    def get_previous_output(self):
        """
        Retrieves previously generated transcription outputs if no new transcription is available
        from the current audio chunks.

        Checks the time since the last transcription output and, if it is within a specified
        threshold, returns the most recent segments of transcribed text. It also manages
        adding a pause (blank segment) to indicate a significant gap in speech based on a defined
        threshold.

        Returns:
            segments (list): A list of transcription segments. This may include the most recent
                            transcribed text segments or a blank segment to indicate a pause
                            in speech.
        """
        segments = []
        if self.t_start is None:
            self.t_start = time.time()

        # Only return incomplete segments that haven't been sent
        if time.time() - self.t_start < self.show_prev_out_thresh:
            segments = self.prepare_segments()

        # add a blank if there is no speech for 3 seconds
        if len(self.text) and self.text[-1] != '':
            if time.time() - self.t_start > self.add_pause_thresh:
                self.text.append('')

        return segments

    def handle_transcription_output(self, transcription_result, audio_duration):
        """
        Handle the transcription output, updating the transcript and sending data to the client.

        Args:
            transcription_result: The result from whisper inference i.e. the list of segments.
            audio_duration (float): Duration of the transcribed audio chunk.
        """
        segments = []
        if len(transcription_result):
            self.t_start = None
            last_segment = self.update_segments(transcription_result, audio_duration)
            segments = self.prepare_segments(last_segment)
        else:
            # show previous output if there is pause i.e. no output from whisper
            segments = self.get_previous_output()

        # Only send if we have segments to send and they're different from the last sent ones
        if len(segments):
            # Only send updates if there are new segments or changes to existing ones
            if self.last_sent_segment is None or len(segments) != len(self.last_sent_segment):
                self.send_transcription_to_client(segments)
                self.last_sent_segment = segments.copy()
            else:
                # Check if any segment's content has changed
                has_changes = False
                for i, segment in enumerate(segments):
                    if i >= len(self.last_sent_segment):
                        has_changes = True
                        break
                    # Check if the text or completed status has changed
                    if (segment.get('text', '') != self.last_sent_segment[i].get('text', '') or
                            segment.get('completed', False) != self.last_sent_segment[i].get('completed', False)):
                        has_changes = True
                        break

                if has_changes:
                    self.send_transcription_to_client(segments)
                    self.last_sent_segment = segments.copy()

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
                continue

            self.clip_audio_if_no_valid_segment()

            input_bytes, duration = self.get_audio_chunk_for_processing()
            if duration < 1.0:
                time.sleep(0.1)  # wait for audio chunks to arrive
                continue
            try:
                input_sample = input_bytes.copy()
                result = self.transcribe_audio(input_sample)

                if result is None or self.language is None:
                    self.timestamp_offset += duration
                    time.sleep(0.25)  # wait for voice activity, result is None when no voice activity
                    continue
                self.handle_transcription_output(result, duration)

            except Exception as e:
                logger.error(f"Failed to transcribe audio chunk: {e}")
                time.sleep(0.01)

    def format_segment(self, start, end, text, completed=False, avg_logprob=None):
        """
        Formats a transcription segment with precise start and end times alongside the transcribed text.

        Args:
            start (float): The start time of the transcription segment in seconds.
            end (float): The end time of the transcription segment in seconds.
            text (str): The transcribed text corresponding to the segment.
            completed (bool): Whether this segment is complete or may be updated.
            avg_logprob (float, optional): The average log probability of tokens, serves as confidence score.

        Returns:
            dict: A dictionary representing the formatted transcription segment, including
                'start' and 'end' times as strings with three decimal places, the 'text'
                of the transcription, whether it's 'completed', and the confidence score (0-1).
        """
        segment = {
            'start': "{:.3f}".format(start),
            'end': "{:.3f}".format(end),
            'text': text,
            'completed': completed
        }

        # Include confidence score if available
        if avg_logprob is not None:
            # Convert log probability to a 0-1 confidence score using exp function
            # Since log probabilities are <= 0, exp(log_prob) will be between 0 and 1
            confidence = math.exp(avg_logprob)
            segment['confidence'] = round(confidence, 4)  # Round to 4 decimal places for readability

        return segment

    def update_segments(self, segments, duration):
        """
        Processes the segments from whisper. Appends all the segments to the list
        except for the last segment assuming that it is incomplete.

        Updates the ongoing transcript with transcribed segments, including their start and end times.
        Complete segments are appended to the transcript in chronological order. Incomplete segments
        (assumed to be the last one) are processed to identify repeated content. If the same incomplete
        segment is seen multiple times, it updates the offset and appends the segment to the transcript.
        A threshold is used to detect repeated content and ensure it is only included once in the transcript.
        The timestamp offset is updated based on the duration of processed segments. The method returns the
        last processed segment, allowing it to be sent to the client for real-time updates.

        Args:
            segments(dict) : dictionary of segments as returned by whisper
            duration(float): duration of the current chunk

        Returns:
            dict or None: The last processed segment with its start time, end time, and transcribed text.
                     Returns None if there are no valid segments to process.
        """
        offset = None
        self.current_out = ''
        last_segment = None

        # process complete segments
        if len(segments) > 1 and segments[-1].no_speech_prob <= self.no_speech_thresh:
            for i, s in enumerate(segments[:-1]):
                text_ = s.text
                self.text.append(text_)
                with self.lock:
                    start, end = self.timestamp_offset + s.start, self.timestamp_offset + min(duration, s.end)

                if start >= end:
                    continue
                if s.no_speech_prob > self.no_speech_thresh:
                    continue

                self.transcript.append(self.format_segment(
                    start,
                    end,
                    text_,
                    completed=True,
                    avg_logprob=s.avg_logprob
                ))
                offset = min(duration, s.end)

        # only process the last segment if it satisfies the no_speech_thresh
        if segments[-1].no_speech_prob <= self.no_speech_thresh:
            self.current_out += segments[-1].text
            # Store the avg_logprob of the last segment for later use
            self.current_segment_avg_logprob = segments[-1].avg_logprob
            with self.lock:
                last_segment = self.format_segment(
                    self.timestamp_offset + segments[-1].start,
                    self.timestamp_offset + min(duration, segments[-1].end),
                    self.current_out,
                    completed=False,
                    avg_logprob=segments[-1].avg_logprob
                )

        if self.current_out.strip() == self.prev_out.strip() and self.current_out != '':
            self.same_output_count += 1

            # if we remove the audio because of same output on the nth reptition we might remove the
            # audio thats not yet transcribed so, capturing the time when it was repeated for the first time
            if self.end_time_for_same_output is None:
                self.end_time_for_same_output = segments[-1].end
            time.sleep(
                0.1)  # wait for some voice activity just in case there is an unitended pause from the speaker for better punctuations.
        else:
            self.same_output_count = 0
            self.end_time_for_same_output = None

        # if same incomplete segment is seen multiple times then update the offset
        # and append the segment to the list
        if self.same_output_count > self.same_output_threshold:
            if not len(self.text) or self.text[-1].strip().lower() != self.current_out.strip().lower():
                self.text.append(self.current_out)
                with self.lock:
                    # Use the stored avg_logprob if available, otherwise use a default
                    avg_logprob = self.current_segment_avg_logprob if self.current_segment_avg_logprob is not None else -1.0
                    self.transcript.append(self.format_segment(
                        self.timestamp_offset,
                        self.timestamp_offset + min(duration, self.end_time_for_same_output),
                        self.current_out,
                        completed=True,
                        avg_logprob=avg_logprob  # Use the actual avg_logprob
                    ))
            self.current_out = ''
            offset = min(duration, self.end_time_for_same_output)
            self.same_output_count = 0
            last_segment = None
            self.end_time_for_same_output = None
        else:
            self.prev_out = self.current_out

        # update offset
        if offset is not None:
            with self.lock:
                self.timestamp_offset += offset

        return last_segment