import functools
import json
import os
import signal
import threading
import time
from typing import Optional

import numpy as np
import torch
from loguru import logger
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from whisper_live.backend_type import BackendType
from whisper_live.client_manager import ClientManager
from whisper_live.serve_client_base import ServeClientBase
from whisper_live.serve_client_faster_whisper import ServeClientFasterWhisper
from whisper_live.vad import VoiceActivityDetector

try:
    from whisper_live.serve_client_tensor_rt import ServeClientTensorRT
    from whisper_live.transcriber_tensorrt import WhisperTRTLLM
except Exception:
    pass


class TranscriptionServer:
    RATE = 16000

    def __init__(self):
        self.client_manager = ClientManager()
        self.no_voice_activity_chunks = 0
        self.use_vad = True
        self.single_model = False
        self.backend = BackendType.FASTER_WHISPER  # Default backend
        self.preloaded_model = False
        self.server = None

    def preload_model(self, backend, faster_whisper_custom_model_path=None, whisper_tensorrt_path=None,
                      trt_multilingual=False, model_name="large-v3-turbo"):
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
                logger.info(f"Preloading faster_whisper model: {model_name}")

                # Model path handling
                model_path = model_name
                if faster_whisper_custom_model_path is not None and os.path.exists(faster_whisper_custom_model_path):
                    logger.info(f"Using custom model path: {faster_whisper_custom_model_path}")
                    model_path = faster_whisper_custom_model_path

                # Determine compute type
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "int8"
                if device == "cuda":
                    major, _ = torch.cuda.get_device_capability(device)
                    compute_type = "float16" if major >= 7 else "float32"

                logger.info(f"Using Device={device} with precision {compute_type}")

                # Create the model
                from whisper_live.whisper_model import WhisperModel
                ServeClientFasterWhisper.SINGLE_MODEL = WhisperModel(
                    model_path,
                    device=device,
                    compute_type=compute_type,
                    local_files_only=False,
                )
                logger.info("Faster whisper model preloaded successfully!")
                self.preloaded_model = True
            except Exception as e:
                logger.error(f"Failed to preload faster_whisper model: {e}")

        elif self.backend.is_tensorrt():
            try:
                logger.info("Preloading TensorRT model")
                if whisper_tensorrt_path is None:
                    logger.error("TensorRT model path not provided, can't preload")
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
                logger.info("Warming up TensorRT engine...")
                mel, _ = ServeClientTensorRT.SINGLE_MODEL.log_mel_spectrogram("assets/jfk.flac")
                for i in range(10):  # 10 warmup steps
                    ServeClientTensorRT.SINGLE_MODEL.transcribe(mel)

                logger.info("TensorRT model preloaded and warmed up successfully!")
                self.preloaded_model = True
            except Exception as e:
                logger.error(f"Failed to preload TensorRT model: {e}")

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
                logger.info("Running TensorRT backend.")
            except Exception as e:
                logger.error(f"TensorRT-LLM not supported: {e}")
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
                    logger.info(f"Using custom model {faster_whisper_custom_model_path}")
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

                logger.info("Running faster_whisper backend.")
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
            logger.info("New client connected")
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
            logger.error("Failed to decode JSON from client")
            return False
        except ConnectionClosed:
            logger.info("Connection closed by client")
            return False
        except Exception as e:
            logger.error(f"Error during new connection initialization: {str(e)}")
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
            logger.info("Connection closed by client")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
        finally:
            if self.client_manager.get_client(websocket):
                self.cleanup(websocket)
                websocket.close()
            del websocket

    def shutdown(self, signum, frame):
        logger.info("Shutting down server...")
        if self.server:
            with self.server as server:
                server.shutdown()
        exit(0)

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

        def handle_shutdown_signal(signum, frame):
            threading.Thread(target=self.shutdown).start()

        signal.signal(signal.SIGINT, handle_shutdown_signal)

        if faster_whisper_custom_model_path is not None and not os.path.exists(faster_whisper_custom_model_path):
            raise ValueError(f"Custom faster_whisper model '{faster_whisper_custom_model_path}' is not a valid path.")
        if whisper_tensorrt_path is not None and not os.path.exists(whisper_tensorrt_path):
            raise ValueError(f"TensorRT model '{whisper_tensorrt_path}' is not a valid path.")
        if single_model:
            if faster_whisper_custom_model_path or whisper_tensorrt_path or backend == "faster_whisper":
                logger.info("Single model mode enabled. Preloading model...")
                self.single_model = True
                if not BackendType.is_valid(backend):
                    raise ValueError(
                        f"{backend} is not a valid backend type. Choose backend from {BackendType.valid_types()}")
                self.preload_model(
                    BackendType(backend),
                    faster_whisper_custom_model_path=faster_whisper_custom_model_path,
                    whisper_tensorrt_path=whisper_tensorrt_path,
                    trt_multilingual=trt_multilingual,
                    model_name="large-v3-turbo"
                )
            else:
                logger.info("Single model mode currently only works with custom models or faster_whisper backend.")
        if not BackendType.is_valid(backend):
            raise ValueError(f"{backend} is not a valid backend type. Choose backend from {BackendType.valid_types()}")
        self.server = serve(
            functools.partial(
                self.recv_audio,
                backend=BackendType(backend),
                faster_whisper_custom_model_path=faster_whisper_custom_model_path,
                whisper_tensorrt_path=whisper_tensorrt_path,
                trt_multilingual=trt_multilingual
            ),
            host,
            port
        )
        with self.server as server:
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
                time.sleep(0.1)  # Sleep 100m; wait some voice activity.
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
