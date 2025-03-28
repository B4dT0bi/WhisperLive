import json
import threading
import numpy as np
from loguru import logger


class ServeClientBase(object):
    RATE = 16000
    SERVER_READY = "SERVER_READY"
    DISCONNECT = "DISCONNECT"

    def __init__(self, client_uid, websocket):
        self.client_uid = client_uid
        self.websocket = websocket
        self.frames = b""
        self.timestamp_offset = 0.0
        self.frames_np = np.array([], dtype=np.float32)
        self.frames_offset = 0.0
        self.text = []
        self.current_out = ''
        self.prev_out = ''
        self.t_start = None
        self.exit = False
        self.same_output_count = 0
        self.show_prev_out_thresh = 2  # if pause(no output from whisper) show previous output for 5 seconds
        self.add_pause_thresh = 3  # add a blank to segment list as a pause(no speech) for 3 seconds
        self.transcript = []
        self.send_last_n_segments = 10

        # Add attributes needed by derived classes
        self.eos = False

        # text formatting
        self.pick_previous_segments = 2

        # threading
        self.lock = threading.Lock()

        # Track already sent completed segments
        self.sent_completed_segments = set()

    def speech_to_text(self):
        raise NotImplementedError

    def transcribe_audio(self, audio_input):
        """
        Transcribe audio input.

        Args:
            audio_input: Audio data to transcribe
        """
        raise NotImplementedError

    def handle_transcription_output(self, transcription_result, audio_duration):
        """
        Handle the transcription output, updating the transcript and sending data to the client.

        Args:
            transcription_result: The result from transcription (can be segments or text)
            audio_duration (float): Duration of the transcribed audio chunk.
        """
        raise NotImplementedError

    def update_timestamp_offset(self, last_segment, duration):
        """
        Update timestamp offset and transcript.

        Args:
            last_segment (str): Last transcribed audio from the whisper model.
            duration (float): Duration of the last audio chunk.
        """
        if not len(self.transcript):
            self.transcript.append(
                {"text": last_segment + " ", "confidence": 0.0})  # TensorRT doesn't provide confidence directly
        elif self.transcript[-1]["text"].strip() != last_segment:
            self.transcript.append(
                {"text": last_segment + " ", "confidence": 0.0})  # TensorRT doesn't provide confidence directly

        with self.lock:
            self.timestamp_offset += duration

    def add_frames(self, frame_np):
        """
        Add audio frames to the ongoing audio stream buffer.

        This method is responsible for maintaining the audio stream buffer, allowing the continuous addition
        of audio frames as they are received. It also ensures that the buffer does not exceed a specified size
        to prevent excessive memory usage.

        If the buffer size exceeds a threshold (45 seconds of audio data), it discards the oldest 30 seconds
        of audio data to maintain a reasonable buffer size. If the buffer is empty, it initializes it with the provided
        audio frame. The audio stream buffer is used for real-time processing of audio data for transcription.

        Args:
            frame_np (numpy.ndarray): The audio frame data as a NumPy array.

        """
        self.lock.acquire()
        if self.frames_np.shape[0] > 45 * self.RATE:
            self.frames_offset += 30.0
            self.frames_np = self.frames_np[int(30 * self.RATE):]
            # check timestamp offset(should be >= self.frame_offset)
            # this basically means that there is no speech as timestamp offset hasnt updated
            # and is less than frame_offset
            if self.timestamp_offset < self.frames_offset:
                self.timestamp_offset = self.frames_offset
        if self.frames_np.shape[0] == 0:
            self.frames_np = frame_np.copy()
        else:
            self.frames_np = np.concatenate((self.frames_np, frame_np), axis=0)
        self.lock.release()

    def clip_audio_if_no_valid_segment(self):
        """
        Update the timestamp offset based on audio buffer status.
        Clip audio if the current chunk exceeds 30 seconds, this basically implies that
        no valid segment for the last 30 seconds from whisper
        """
        with self.lock:
            if self.frames_np[int((self.timestamp_offset - self.frames_offset) * self.RATE):].shape[0] > 25 * self.RATE:
                duration = self.frames_np.shape[0] / self.RATE
                self.timestamp_offset = self.frames_offset + duration - 5

    def get_audio_chunk_for_processing(self):
        """
        Retrieves the next chunk of audio data for processing based on the current offsets.

        Calculates which part of the audio data should be processed next, based on
        the difference between the current timestamp offset and the frame's offset, scaled by
        the audio sample rate (RATE). It then returns this chunk of audio data along with its
        duration in seconds.

        Returns:
            tuple: A tuple containing:
                - input_bytes (np.ndarray): The next chunk of audio data to be processed.
                - duration (float): The duration of the audio chunk in seconds.
        """
        with self.lock:
            samples_take = max(0, (self.timestamp_offset - self.frames_offset) * self.RATE)
            input_bytes = self.frames_np[int(samples_take):].copy()
        duration = input_bytes.shape[0] / self.RATE
        return input_bytes, duration

    def prepare_segments(self, last_segment=None):
        """
        Prepares the segments of transcribed text to be sent to the client.

        This method compiles only segments that haven't been sent before or are incomplete.
        Once a segment is marked as completed and sent, it won't be included in future responses.

        Args:
            last_segment (str, optional): The most recent segment of transcribed text to be added
                                          to the list of segments. Defaults to None.

        Returns:
            list: A list of transcribed text segments to be sent to the client.
        """
        segments = []

        if len(self.transcript) >= self.send_last_n_segments:
            segments = self.transcript[-self.send_last_n_segments:].copy()
        else:
            segments = self.transcript.copy()

        if last_segment is not None:
            segments = segments + [last_segment]

        return segments

    def get_audio_chunk_duration(self, input_bytes):
        """
        Calculates the duration of the provided audio chunk.

        Args:
            input_bytes (numpy.ndarray): The audio chunk for which to calculate the duration.

        Returns:
            float: The duration of the audio chunk in seconds.
        """
        return input_bytes.shape[0] / self.RATE

    def send_transcription_to_client(self, segments):
        """
        Sends the specified transcription segments to the client over the websocket connection.

        This method formats the transcription segments into a JSON object and attempts to send
        this object to the client. If an error occurs during the send operation, it logs the error.

        Only segments that haven't been sent before will be included in the response.
        Completed segments are tracked to avoid duplicate sending.

        Args:
            segments (list): A list of transcription segments to be sent to the client.
        """
        try:
            # Filter out completed segments that have already been sent
            filtered_segments = []
            logger.info(f"Segments before filtering: {segments}")
            for segment in segments:
                # Create a unique ID for the segment based on its content and timestamps
                segment_id = None
                if segment.get('completed', False) and 'text' in segment and 'start' in segment and 'end' in segment:
                    segment_id = f"{segment['text']}_{segment['start']}_{segment['end']}"

                # Only include segments that are not completed or haven't been sent before
                if not segment.get('completed', False) or segment_id not in self.sent_completed_segments:
                    filtered_segments.append(segment)
                    # Track completed segments to avoid sending them again
                    if segment.get('completed', False) and segment_id is not None:
                        self.sent_completed_segments.add(segment_id)

            # Only send the response if there are segments to send
            if filtered_segments:
                response = {
                    "uid": self.client_uid,
                    "segments": filtered_segments,
                }
                logger.info(f"Sending response to client {self.client_uid}: {json.dumps(response, indent=2)}")
                self.websocket.send(json.dumps(response))
        except Exception as e:
            logger.error(f"Sending data to client: {e}")

    def disconnect(self):
        """
        Notify the client of disconnection and send a disconnect message.

        This method sends a disconnect message to the client via the WebSocket connection to notify them
        that the transcription service is disconnecting gracefully.

        """
        self.websocket.send(json.dumps({
            "uid": self.client_uid,
            "message": self.DISCONNECT
        }))

    def cleanup(self):
        """
        Perform cleanup tasks before exiting the transcription service.

        This method performs necessary cleanup tasks, including stopping the transcription thread, marking
        the exit flag to indicate the transcription thread should exit gracefully, and destroying resources
        associated with the transcription process.

        """
        logger.info("Cleaning up.")
        self.exit = True
