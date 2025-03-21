import os
import time
import threading
import json
import functools
import logging
import math
from enum import Enum
from typing import List, Optional

import torch
import numpy as np
from websockets.sync.server import serve
from websockets.exceptions import ConnectionClosed
from whisper_live.vad import VoiceActivityDetector
from whisper_live.transcriber import WhisperModel
try:
    from whisper_live.transcriber_tensorrt import WhisperTRTLLM
except Exception:
    pass

logging.basicConfig(level=logging.INFO)


class ClientManager:
    def __init__(self, max_clients=4, max_connection_time=600):
        """
        Initializes the ClientManager with specified limits on client connections and connection durations.

        Args:
            max_clients (int, optional): The maximum number of simultaneous client connections allowed. Defaults to 4.
            max_connection_time (int, optional): The maximum duration (in seconds) a client can stay connected. Defaults
                                                 to 600 seconds (10 minutes).
        """
        self.clients = {}
        self.start_times = {}
        self.max_clients = max_clients
        self.max_connection_time = max_connection_time

    def add_client(self, websocket, client):
        """
        Adds a client and their connection start time to the tracking dictionaries.

        Args:
            websocket: The websocket associated with the client to add.
            client: The client object to be added and tracked.
        """
        self.clients[websocket] = client
        self.start_times[websocket] = time.time()

    def get_client(self, websocket):
        """
        Retrieves a client associated with the given websocket.

        Args:
            websocket: The websocket associated with the client to retrieve.

        Returns:
            The client object if found, False otherwise.
        """
        if websocket in self.clients:
            return self.clients[websocket]
        return False

    def remove_client(self, websocket):
        """
        Removes a client and their connection start time from the tracking dictionaries. Performs cleanup on the
        client if necessary.

        Args:
            websocket: The websocket associated with the client to be removed.
        """
        client = self.clients.pop(websocket, None)
        if client:
            client.cleanup()
        self.start_times.pop(websocket, None)

    def get_wait_time(self):
        """
        Calculates the estimated wait time for new clients based on the remaining connection times of current clients.

        Returns:
            The estimated wait time in minutes for new clients to connect. Returns 0 if there are available slots.
        """
        wait_time = None
        for start_time in self.start_times.values():
            current_client_time_remaining = self.max_connection_time - (time.time() - start_time)
            if wait_time is None or current_client_time_remaining < wait_time:
                wait_time = current_client_time_remaining
        return wait_time / 60 if wait_time is not None else 0

    def is_server_full(self, websocket, options):
        """
        Checks if the server is at its maximum client capacity and sends a wait message to the client if necessary.

        Args:
            websocket: The websocket of the client attempting to connect.
            options: A dictionary of options that may include the client's unique identifier.

        Returns:
            True if the server is full, False otherwise.
        """
        if len(self.clients) >= self.max_clients:
            wait_time = self.get_wait_time()
            response = {"uid": options["uid"], "status": "WAIT", "message": wait_time}
            websocket.send(json.dumps(response))
            return True
        return False

    def is_client_timeout(self, websocket):
        """
        Checks if a client has exceeded the maximum allowed connection time and disconnects them if so, issuing a warning.

        Args:
            websocket: The websocket associated with the client to check.

        Returns:
            True if the client's connection time has exceeded the maximum limit, False otherwise.
        """
        elapsed_time = time.time() - self.start_times[websocket]
        if elapsed_time >= self.max_connection_time:
            self.clients[websocket].disconnect()
            logging.warning(f"Client with uid '{self.clients[websocket].client_uid}' disconnected due to overtime.")
            return True
        return False


class BackendType(Enum):
    FASTER_WHISPER = "faster_whisper"
    TENSORRT = "tensorrt"

    @staticmethod
    def valid_types() -> List[str]:
        return [backend_type.value for backend_type in BackendType]

    @staticmethod
    def is_valid(backend: str) -> bool:
        return backend in BackendType.valid_types()

    def is_faster_whisper(self) -> bool:
        return self == BackendType.FASTER_WHISPER

    def is_tensorrt(self) -> bool:
        return self == BackendType.TENSORRT


class TranscriptionServer:
    RATE = 16000

    def __init__(self):
        self.client_manager = ClientManager()
        self.no_voice_activity_chunks = 0
        self.use_vad = True
        self.single_model = False
        self.backend = BackendType.FASTER_WHISPER  # Default backend
        self.preloaded_model = False

    def preload_model(self, backend, faster_whisper_custom_model_path=None, whisper_tensorrt_path=None, trt_multilingual=False, model_name="large-v3-turbo"):
        """
        Preload the model at server startup to avoid delay on first connection.
        
        Args:
            backend (BackendType): The backend to use (faster_whisper or tensorrt)
            faster_whisper_custom_model_path (str, optional): Path to custom faster whisper model.
            whisper_tensorrt_path (str, optional): Path to tensorrt model.
            trt_multilingual (bool, optional): Whether to use multilingual model (for TensorRT).
            model_name (str, optional): Name of the model to preload for faster_whisper. Default is "large-v3-turbo".
        """
        self.backend = backend
        if self.backend.is_faster_whisper():
            try:
                logging.info(f"Preloading faster_whisper model: {model_name}")
                
                # Model path handling
                model_path = model_name
                if faster_whisper_custom_model_path is not None and os.path.exists(faster_whisper_custom_model_path):
                    logging.info(f"Using custom model path: {faster_whisper_custom_model_path}")
                    model_path = faster_whisper_custom_model_path
                
                # Determine compute type
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "int8"
                if device == "cuda":
                    major, _ = torch.cuda.get_device_capability(device)
                    compute_type = "float16" if major >= 7 else "float32"
                
                logging.info(f"Using Device={device} with precision {compute_type}")
                
                # Create the model
                from whisper_live.transcriber import WhisperModel
                ServeClientFasterWhisper.SINGLE_MODEL = WhisperModel(
                    model_path,
                    device=device,
                    compute_type=compute_type,
                    local_files_only=False,
                )
                logging.info("Faster whisper model preloaded successfully!")
                self.preloaded_model = True
            except Exception as e:
                logging.error(f"Failed to preload faster_whisper model: {e}")
        
        elif self.backend.is_tensorrt():
            try:
                logging.info("Preloading TensorRT model")
                if whisper_tensorrt_path is None:
                    logging.error("TensorRT model path not provided, can't preload")
                    return
                
                # Preload TensorRT model
                from whisper_live.transcriber_tensorrt import WhisperTRTLLM
                ServeClientTensorRT.SINGLE_MODEL = WhisperTRTLLM(
                    whisper_tensorrt_path,
                    assets_dir="assets",
                    device="cuda",
                    is_multilingual=trt_multilingual,
                    language="en",
                    task="transcribe"
                )
                
                # Warmup
                logging.info("Warming up TensorRT engine...")
                mel, _ = ServeClientTensorRT.SINGLE_MODEL.log_mel_spectrogram("assets/jfk.flac")
                for i in range(10):  # 10 warmup steps
                    ServeClientTensorRT.SINGLE_MODEL.transcribe(mel)
                
                logging.info("TensorRT model preloaded and warmed up successfully!")
                self.preloaded_model = True
            except Exception as e:
                logging.error(f"Failed to preload TensorRT model: {e}")

    def initialize_client(
        self, websocket, options, faster_whisper_custom_model_path,
        whisper_tensorrt_path, trt_multilingual
    ):
        client: Optional[ServeClientBase] = None

        if self.backend.is_tensorrt():
            try:
                client = ServeClientTensorRT(
                    websocket,
                    multilingual=trt_multilingual,
                    language=options["language"],
                    task=options["task"],
                    client_uid=options["uid"],
                    model=whisper_tensorrt_path,
                    single_model=self.single_model,
                )
                logging.info("Running TensorRT backend.")
            except Exception as e:
                logging.error(f"TensorRT-LLM not supported: {e}")
                self.client_uid = options["uid"]
                websocket.send(json.dumps({
                    "uid": self.client_uid,
                    "status": "WARNING",
                    "message": "TensorRT-LLM not supported on Server yet. "
                               "Reverting to available backend: 'faster_whisper'"
                }))
                self.backend = BackendType.FASTER_WHISPER

        try:
            if self.backend.is_faster_whisper():
                if faster_whisper_custom_model_path is not None and os.path.exists(faster_whisper_custom_model_path):
                    logging.info(f"Using custom model {faster_whisper_custom_model_path}")
                    options["model"] = faster_whisper_custom_model_path
                client = ServeClientFasterWhisper(
                    websocket,
                    language=options["language"],
                    task=options["task"],
                    client_uid=options["uid"],
                    model=options["model"],
                    initial_prompt=options.get("initial_prompt"),
                    vad_parameters=options.get("vad_parameters"),
                    use_vad=self.use_vad,
                    single_model=self.single_model,
                )

                logging.info("Running faster_whisper backend.")
        except Exception as e:
            return

        if client is None:
            raise ValueError(f"Backend type {self.backend.value} not recognised or not handled.")

        self.client_manager.add_client(websocket, client)

    def get_audio_from_websocket(self, websocket):
        """
        Receives audio buffer from websocket and creates a numpy array out of it.

        Args:
            websocket: The websocket to receive audio from.

        Returns:
            A numpy array containing the audio.
        """
        frame_data = websocket.recv()
        if frame_data == b"END_OF_AUDIO":
            return False
        return np.frombuffer(frame_data, dtype=np.float32)

    def handle_new_connection(self, websocket, faster_whisper_custom_model_path,
                              whisper_tensorrt_path, trt_multilingual):
        try:
            logging.info("New client connected")
            options = websocket.recv()
            options = json.loads(options)

            if self.client_manager is None:
                max_clients = options.get('max_clients', 4)
                max_connection_time = options.get('max_connection_time', 600)
                self.client_manager = ClientManager(max_clients, max_connection_time)

            self.use_vad = options.get('use_vad')
            if self.client_manager.is_server_full(websocket, options):
                websocket.close()
                return False  # Indicates that the connection should not continue

            if self.backend.is_tensorrt():
                self.vad_detector = VoiceActivityDetector(frame_rate=self.RATE)
            self.initialize_client(websocket, options, faster_whisper_custom_model_path,
                                   whisper_tensorrt_path, trt_multilingual)
            return True
        except json.JSONDecodeError:
            logging.error("Failed to decode JSON from client")
            return False
        except ConnectionClosed:
            logging.info("Connection closed by client")
            return False
        except Exception as e:
            logging.error(f"Error during new connection initialization: {str(e)}")
            return False

    def process_audio_frames(self, websocket):
        frame_np = self.get_audio_from_websocket(websocket)
        client = self.client_manager.get_client(websocket)
        if client is False:
            return False
            
        if frame_np is False:
            if self.backend.is_tensorrt():
                client.set_eos(True)
            return False

        if self.backend.is_tensorrt():
            voice_active = self.voice_activity(websocket, frame_np)
            if voice_active:
                self.no_voice_activity_chunks = 0
                client.set_eos(False)
            if self.use_vad and not voice_active:
                return True

        client.add_frames(frame_np)
        return True

    def recv_audio(self,
                   websocket,
                   backend: BackendType = BackendType.FASTER_WHISPER,
                   faster_whisper_custom_model_path=None,
                   whisper_tensorrt_path=None,
                   trt_multilingual=False):
        """
        Receive audio chunks from a client in an infinite loop.

        Continuously receives audio frames from a connected client
        over a WebSocket connection. It processes the audio frames using a
        voice activity detection (VAD) model to determine if they contain speech
        or not. If the audio frame contains speech, it is added to the client's
        audio data for ASR.
        If the maximum number of clients is reached, the method sends a
        "WAIT" status to the client, indicating that they should wait
        until a slot is available.
        If a client's connection exceeds the maximum allowed time, it will
        be disconnected, and the client's resources will be cleaned up.

        Args:
            websocket (WebSocket): The WebSocket connection for the client.
            backend (str): The backend to run the server with.
            faster_whisper_custom_model_path (str): path to custom faster whisper model.
            whisper_tensorrt_path (str): Required for tensorrt backend.
            trt_multilingual(bool): Only used for tensorrt, True if multilingual model.

        Raises:
            Exception: If there is an error during the audio frame processing.
        """
        self.backend = backend
        if not self.handle_new_connection(websocket, faster_whisper_custom_model_path,
                                          whisper_tensorrt_path, trt_multilingual):
            return

        try:
            while not self.client_manager.is_client_timeout(websocket):
                if not self.process_audio_frames(websocket):
                    break
        except ConnectionClosed:
            logging.info("Connection closed by client")
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
        finally:
            if self.client_manager.get_client(websocket):
                self.cleanup(websocket)
                websocket.close()
            del websocket

    def run(self,
            host,
            port=9090,
            backend="tensorrt",
            faster_whisper_custom_model_path=None,
            whisper_tensorrt_path=None,
            trt_multilingual=False,
            single_model=False):
        """
        Run the transcription server.

        Args:
            host (str): The host address to bind the server.
            port (int): The port number to bind the server.
        """
        if faster_whisper_custom_model_path is not None and not os.path.exists(faster_whisper_custom_model_path):
            raise ValueError(f"Custom faster_whisper model '{faster_whisper_custom_model_path}' is not a valid path.")
        if whisper_tensorrt_path is not None and not os.path.exists(whisper_tensorrt_path):
            raise ValueError(f"TensorRT model '{whisper_tensorrt_path}' is not a valid path.")
        if single_model:
            if faster_whisper_custom_model_path or whisper_tensorrt_path or backend == "faster_whisper":
                logging.info("Single model mode enabled. Preloading model...")
                self.single_model = True
                if not BackendType.is_valid(backend):
                    raise ValueError(f"{backend} is not a valid backend type. Choose backend from {BackendType.valid_types()}")
                self.preload_model(
                    BackendType(backend),
                    faster_whisper_custom_model_path=faster_whisper_custom_model_path,
                    whisper_tensorrt_path=whisper_tensorrt_path,
                    trt_multilingual=trt_multilingual,
                    model_name="large-v3-turbo"
                )
            else:
                logging.info("Single model mode currently only works with custom models or faster_whisper backend.")
        if not BackendType.is_valid(backend):
            raise ValueError(f"{backend} is not a valid backend type. Choose backend from {BackendType.valid_types()}")
        with serve(
            functools.partial(
                self.recv_audio,
                backend=BackendType(backend),
                faster_whisper_custom_model_path=faster_whisper_custom_model_path,
                whisper_tensorrt_path=whisper_tensorrt_path,
                trt_multilingual=trt_multilingual
            ),
            host,
            port
        ) as server:
            server.serve_forever()

    def voice_activity(self, websocket, frame_np):
        """
        Evaluates the voice activity in a given audio frame and manages the state of voice activity detection.

        This method uses the configured voice activity detection (VAD) model to assess whether the given audio frame
        contains speech. If the VAD model detects no voice activity for more than three consecutive frames,
        it sets an end-of-speech (EOS) flag for the associated client. This method aims to efficiently manage
        speech detection to improve subsequent processing steps.

        Args:
            websocket: The websocket associated with the current client. Used to retrieve the client object
                    from the client manager for state management.
            frame_np (numpy.ndarray): The audio frame to be analyzed. This should be a NumPy array containing
                                    the audio data for the current frame.

        Returns:
            bool: True if voice activity is detected in the current frame, False otherwise. When returning False
                after detecting no voice activity for more than three consecutive frames, it also triggers the
                end-of-speech (EOS) flag for the client.
        """
        if not self.vad_detector(frame_np):
            self.no_voice_activity_chunks += 1
            if self.no_voice_activity_chunks > 3:
                client = self.client_manager.get_client(websocket)
                if client is not False and not client.eos:
                    client.set_eos(True)
                time.sleep(0.1)    # Sleep 100m; wait some voice activity.
            return False
        return True

    def cleanup(self, websocket):
        """
        Cleans up resources associated with a given client's websocket.

        Args:
            websocket: The websocket associated with the client to be cleaned up.
        """
        if self.client_manager.get_client(websocket):
            self.client_manager.remove_client(websocket)


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
        self.show_prev_out_thresh = 5   # if pause(no output from whisper) show previous output for 5 seconds
        self.add_pause_thresh = 3       # add a blank to segment list as a pause(no speech) for 3 seconds
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
        raise NotImplementedError
        
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
        if self.frames_np.shape[0] > 45*self.RATE:
            self.frames_offset += 30.0
            self.frames_np = self.frames_np[int(30*self.RATE):]
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
            if self.frames_np[int((self.timestamp_offset - self.frames_offset)*self.RATE):].shape[0] > 25 * self.RATE:
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

        Returns:
            segments (list): A list of transcription segments to be sent to the client.
        """
        try:
            response = {
                "uid": self.client_uid,
                "segments": segments,
            }
            logging.info(f"Sending response to client {self.client_uid}: {json.dumps(response, indent=2)}")
            self.websocket.send(json.dumps(response))
        except Exception as e:
            logging.error(f"[ERROR]: Sending data to client: {e}")

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
        logging.info("Cleaning up.")
        self.exit = True


class ServeClientTensorRT(ServeClientBase):

    SINGLE_MODEL = None
    SINGLE_MODEL_LOCK = threading.Lock()

    def __init__(self, websocket, task="transcribe", multilingual=False, language=None, client_uid=None, model=None, single_model=False):
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
                logging.info("Using preloaded TensorRT model")
                self.transcriber = ServeClientTensorRT.SINGLE_MODEL
            else:
                logging.info("TensorRT single model was requested but not preloaded. Creating model...")
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
        logging.info("[INFO:] Warming up TensorRT engine..")
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
        segments = self.prepare_segments({"text": transcription_result, "confidence": 0.0})  # TensorRT doesn't provide confidence directly
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
        logging.info(f"[WhisperTensorRT:] Processing audio with duration: {audio_input.shape[0] / self.RATE}")
        mel, duration = self.transcriber.log_mel_spectrogram(audio_input)
        last_segment = self.transcriber.transcribe(
            mel,
            text_prefix=f"<|startoftranscript|><|{self.language}|><|{self.task}|><|notimestamps|>"
        )
        if ServeClientTensorRT.SINGLE_MODEL:
            ServeClientTensorRT.SINGLE_MODEL_LOCK.release()
        if last_segment:
            self.handle_transcription_output(last_segment, duration)

    def update_timestamp_offset(self, last_segment, duration):
        """
        Update timestamp offset and transcript.
        Args:
            last_segment (str): Last transcribed audio from the whisper model.
            duration (float): Duration of the last audio chunk.
        """
        if not len(self.transcript):
            self.transcript.append({"text": last_segment + " "})
        elif self.transcript[-1]["text"].strip() != last_segment:
            self.transcript.append({"text": last_segment + " "})

        with self.lock:
            self.timestamp_offset += duration

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
                logging.info("Exiting speech to text thread")
                break

            if self.frames_np.shape[0] == 0:
                time.sleep(0.02)    # wait for any audio to arrive
                continue

            self.clip_audio_if_no_valid_segment()

            input_bytes, duration = self.get_audio_chunk_for_processing()
            if duration < 0.4:
                continue

            try:
                input_sample = input_bytes.copy()
                logging.info(f"[WhisperTensorRT:] Processing audio with duration: {duration}")
                self.transcribe_audio(input_sample)

            except Exception as e:
                logging.error(f"[ERROR]: {e}")


class ServeClientFasterWhisper(ServeClientBase):

    SINGLE_MODEL = None
    SINGLE_MODEL_LOCK = threading.Lock()

    def __init__(self, websocket, task="transcribe", device=None, language=None, client_uid=None, model="large-v3-turbo",
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
        self.no_speech_thresh = 0.6
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
        self.no_speech_thresh = 0.45
        self.same_output_threshold = 10
        self.end_time_for_same_output = None

        # Add this to track the last sent segment
        self.last_sent_segment = None

        if self.model_size_or_path is None:
            return
        logging.info(f"Using Device={self.device} with precision {self.compute_type}")
    
        try:
            # Check if we should use the preloaded model
            if single_model:
                if ServeClientFasterWhisper.SINGLE_MODEL is not None:
                    logging.info("Using preloaded single model")
                    self.transcriber = ServeClientFasterWhisper.SINGLE_MODEL
                else:
                    logging.info("Single model was requested but not preloaded. Creating model...")
                    self.create_model(self.device)
                    ServeClientFasterWhisper.SINGLE_MODEL = self.transcriber
            else:
                self.create_model(self.device)
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
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
            logging.info(f"Detected language {self.language} with probability {info.language_probability}")
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
                logging.info("Exiting speech to text thread")
                break

            if self.frames_np.shape[0] == 0:
                continue

            self.clip_audio_if_no_valid_segment()

            input_bytes, duration = self.get_audio_chunk_for_processing()
            if duration < 1.0:
                time.sleep(0.1)     # wait for audio chunks to arrive
                continue
            try:
                input_sample = input_bytes.copy()
                result = self.transcribe_audio(input_sample)

                if result is None or self.language is None:
                    self.timestamp_offset += duration
                    time.sleep(0.25)    # wait for voice activity, result is None when no voice activity
                    continue
                self.handle_transcription_output(result, duration)

            except Exception as e:
                logging.error(f"[ERROR]: Failed to transcribe audio chunk: {e}")
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
            time.sleep(0.1)     # wait for some voice activity just in case there is an unitended pause from the speaker for better punctuations.
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
