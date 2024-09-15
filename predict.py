import os
import sys
import time
import json
import shutil
import logging
from typing import Iterator, Optional
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import subprocess
import torch
import numpy as np
import cv2

from cog import BasePredictor, Input, Path as CogPath
from huggingface_hub import snapshot_download, hf_hub_download

# Add the 'scripts' directory to the Python path
sys.path.insert(0, os.path.abspath("scripts"))

# Import necessary modules from scripts
from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
from musetalk.utils.blending import get_image, get_image_prepare_material, get_image_blending

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
SESSION_CACHE_DIR = "/tmp/data/session"
MAX_SESSIONS_IN_MEMORY = 1  # Number of sessions to keep in memory

# Model download information
downloads = [
    (
        "https://drive.usercontent.google.com/download?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812&export=download",
        "models/face-parse-bisent/79999_iter.pth",
    ),
    (
        "https://download.pytorch.org/models/resnet18-5c106cde.pth",
        "models/face-parse-bisent/resnet18-5c106cde.pth",
    ),
    (
        "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
        "models/whisper/tiny.pt",
    ),
]

downloads_hf = [
    (
        "TMElyralab/MuseTalk",
        "models",
    ),
    (
        "stabilityai/sd-vae-ft-mse",
        "models/sd-vae-ft-mse",
    ),
]

def download_hf_model(name: str, dest: str) -> None:
    """
    Download a model from Hugging Face Hub.

    Args:
        name (str): The name of the model on Hugging Face Hub.
        dest (str): The destination directory to save the model.

    Raises:
        Exception: If there's an error during the download process.
    """
    cache_dir = "./cache"
    try:
        logger.info(f"Downloading model: {name}")
        logger.info(f"Destination: {dest}")
        snapshot_download(
            cache_dir=cache_dir,
            local_dir=dest,
            repo_id=name,
            allow_patterns=["*.json", "*.bin"],
            ignore_patterns=["*.safetensors", "*.msgpack", "*.h5", "*.ot"],
        )
    except Exception as e:
        logger.error(f"Error downloading model {name}: {str(e)}")
        raise

def download_model(url: str, dest: str) -> None:
    """
    Download a model from a given URL.

    Args:
        url (str): The URL to download the model from.
        dest (str): The destination file path to save the model.

    Raises:
        subprocess.CalledProcessError: If the download process fails.
    """
    dest_dir = os.path.dirname(dest)
    os.makedirs(dest_dir, exist_ok=True)

    if not os.path.exists(dest):
        start = time.time()
        logger.info(f"Downloading from: {url}")
        logger.info(f"Saving to: {dest}")
        try:
            subprocess.check_call(["pget", url, dest], close_fds=False)
            logger.info(f"Download completed in {time.time() - start:.2f} seconds")
        except subprocess.CalledProcessError as e:
            logger.error(f"Download failed: {str(e)}")
            raise
    else:
        logger.info(f"File already exists: {dest}")

class Predictor(BasePredictor):
    def setup(self) -> None:
        """
        Load the model into memory to make running multiple predictions efficient.
        This method downloads necessary model weights and sets up the session cache.
        """
        logger.info("Setting up the predictor...")
        
        with ThreadPoolExecutor() as executor:
            # Download model weights
            futures = [
                executor.submit(download_model, url, dest) for url, dest in downloads
            ]
            futures += [
                executor.submit(download_hf_model, name, dest) for name, dest in downloads_hf
            ]
            futures += [
                executor.submit(hf_hub_download, cache_dir="./cache", local_dir="models/dwpose", repo_id="yzd-v/DWPose", filename="dw-ll_ucoco_384.pth")
            ]

            # Wait for all downloads to complete
            for future in futures:
                future.result()

        # Load all models
        self.audio_processor, self.vae, self.unet, self.pe = load_all_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timesteps = torch.tensor([0], device=self.device)

        # Initialize session cache
        self.session_cache = OrderedDict()
        logger.info("Predictor setup completed.")

    def predict(
        self,
        session_id: str = Input(description="Session ID for caching"),
        audio_input: Optional[CogPath] = Input(description="Upload your audio file here.", default=None),
        video_input: Optional[CogPath] = Input(description="Upload your video file here.", default=None),
        bbox_shift: int = Input(description="Bounding box shift for face detection", default=0),
        fps: int = Input(description="Frames per second for output video", default=25),
    ) -> Iterator[CogPath]:
        """
        Run a single prediction on the model.

        This method handles both video preprocessing and audio-based video generation.
        It uses a session-based caching mechanism to improve efficiency for repeated operations.

        Args:
            session_id (str): Unique identifier for the session.
            audio_input (Optional[CogPath]): Path to the input audio file.
            video_input (Optional[CogPath]): Path to the input video file.
            bbox_shift (int): Bounding box shift for face detection.
            fps (int): Frames per second for the output video.

        Yields:
            Iterator[CogPath]: Paths to intermediate results and the final output video.

        Raises:
            ValueError: If neither audio_input nor video_input is provided, or if both are provided.
        """
        logger.info(f"Starting prediction for session: {session_id}")

        if video_input and audio_input:
            logger.error("Both video and audio inputs provided")
            raise ValueError("Provide either video_input or audio_input, not both")

        if video_input:
            logger.info("Processing video input for preprocessing")
            yield from self.precompute_video_data(session_id, str(video_input), bbox_shift)
        elif audio_input:
            logger.info("Generating video from audio input")
            yield from self.generate_video(session_id, str(audio_input), fps, bbox_shift)
        else:
            logger.error("No input provided")
            raise ValueError("Provide either video_input or audio_input")

    def precompute_video_data(self, session_id: str, video_path: str, bbox_shift: int) -> Iterator[str]:
        """
        Precompute and cache video data for future use.

        Args:
            session_id (str): Unique identifier for the session.
            video_path (str): Path to the input video file.
            bbox_shift (int): Bounding box shift for face detection.

        Yields:
            str: Status messages indicating progress.

        Raises:
            Exception: If there's an error during video preprocessing.
        """
        logger.info(f"Starting video preprocessing for session: {session_id}")
        yield "Starting video preprocessing..."

        try:
            session_dir = os.path.join(SESSION_CACHE_DIR, session_id)
            os.makedirs(session_dir, exist_ok=True)

            # Extract frames from video
            if get_file_type(video_path) == "video":
                frame_dir = os.path.join(session_dir, "frames")
                os.makedirs(frame_dir, exist_ok=True)
                cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {frame_dir}/%08d.png"
                subprocess.run(cmd, shell=True, check=True)
                input_img_list = sorted(glob.glob(os.path.join(frame_dir, '*.[jpJP][pnPN]*[gG]')))
                fps = get_video_fps(video_path)
            elif get_file_type(video_path) == "image":
                input_img_list = [video_path]
                fps = 25  # Default fps for single image
            else:
                raise ValueError(f"Unsupported file type: {video_path}")

            yield "Extracting landmarks and bounding boxes..."
            coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)

            yield "Processing frames..."
            input_latent_list = []
            for bbox, frame in zip(coord_list, frame_list):
                if bbox == coord_placeholder:
                    continue
                x1, y1, x2, y2 = bbox
                crop_frame = frame[y1:y2, x1:x2]
                crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                latents = self.vae.get_latents_for_unet(crop_frame)
                input_latent_list.append(latents)

            # Create cyclic lists
            frame_list_cycle = frame_list + frame_list[::-1]
            coord_list_cycle = coord_list + coord_list[::-1]
            input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

            # Save preprocessed data
            preprocessed_data = {
                "frame_list_cycle": frame_list_cycle,
                "coord_list_cycle": coord_list_cycle,
                "input_latent_list_cycle": input_latent_list_cycle,
                "fps": fps
            }
            torch.save(preprocessed_data, os.path.join(session_dir, "preprocessed_data.pt"))

            # Update session cache
            self.session_cache[session_id] = session_dir
            if len(self.session_cache) > MAX_SESSIONS_IN_MEMORY:
                oldest_session = next(iter(self.session_cache))
                del self.session_cache[oldest_session]

            logger.info(f"Video preprocessing completed for session: {session_id}")
            yield json.dumps({"status": "success", "message": "Video data precomputed and stored"})
        except Exception as e:
            logger.error(f"Error during video preprocessing: {str(e)}")
            yield json.dumps({"status": "error", "message": str(e)})
            raise

    def generate_video(self, session_id: str, audio_path: str, fps: int, bbox_shift: int) -> Iterator[CogPath]:
        """
        Generate lip-synced video using precomputed data and audio input.

        Args:
            session_id (str): Unique identifier for the session.
            audio_path (str): Path to the input audio file.
            fps (int): Frames per second for the output video.
            bbox_shift (int): Bounding box shift for face detection.

        Yields:
            Iterator[CogPath]: Paths to intermediate results and the final output video.

        Raises:
            ValueError: If no precomputed data is found for the given session.
            Exception: If there's an error during video generation.
        """
        logger.info(f"Starting video generation for session: {session_id}")
        yield "Starting video generation..."

        try:
            if session_id not in self.session_cache:
                session_path = os.path.join(SESSION_CACHE_DIR, session_id)
                if not os.path.exists(session_path):
                    logger.error(f"No precomputed data found for session: {session_id}")
                    raise ValueError(f"No precomputed data found for session {session_id}")
                self.session_cache[session_id] = session_path

            # Load preprocessed data
            preprocessed_data = torch.load(os.path.join(self.session_cache[session_id], "preprocessed_data.pt"))
            frame_list_cycle = preprocessed_data["frame_list_cycle"]
            coord_list_cycle = preprocessed_data["coord_list_cycle"]
            input_latent_list_cycle = preprocessed_data["input_latent_list_cycle"]
            video_fps = preprocessed_data["fps"]

            # Process audio
            yield "Processing audio..."
            whisper_feature = self.audio_processor.audio2feat(audio_path)
            whisper_chunks = self.audio_processor.feature2chunks(feature_array=whisper_feature, fps=video_fps)

            # Generate video frames
            yield "Generating video frames..."
            output_dir = os.path.join(self.session_cache[session_id], "output")
            os.makedirs(output_dir, exist_ok=True)

            video_num = len(whisper_chunks)
            batch_size = 8  # You can adjust this based on your GPU memory
            gen = datagen(whisper_chunks, input_latent_list_cycle, batch_size)

            for i, (whisper_batch, latent_batch) in enumerate(gen):
                yield f"Processing batch {i+1}/{len(whisper_chunks)//batch_size + 1}"
                audio_feature_batch = torch.from_numpy(whisper_batch).to(device=self.device, dtype=self.unet.model.dtype)
                audio_feature_batch = self.pe(audio_feature_batch)
                latent_batch = latent_batch.to(dtype=self.unet.model.dtype)

                pred_latents = self.unet.model(latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch).sample
                recon = self.vae.decode_latents(pred_latents)

                for j, res_frame in enumerate(recon):
                    frame_index = i * batch_size + j
                    if frame_index >= video_num:
                        break
                    bbox = coord_list_cycle[frame_index % len(coord_list_cycle)]
                    ori_frame = frame_list_cycle[frame_index % len(frame_list_cycle)].copy()
                    x1, y1, x2, y2 = bbox
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
                    combine_frame = get_image(ori_frame, res_frame, bbox)
                    cv2.imwrite(os.path.join(output_dir, f"{frame_index:08d}.png"), combine_frame)

            # Create video from frames
            yield "Creating final video..."
            output_video = os.path.join(self.session_cache[session_id], "output_video.mp4")
            cmd = f"ffmpeg -y -v warning -r {video_fps} -i {output_dir}/%08d.png -c:v libx264 -preset slow -crf 22 -pix_fmt yuv420p {output_video}"
            subprocess.run(cmd, shell=True, check=True)

            # Add audio to video
            yield "Adding audio to video..."
            final_output = os.path.join(self.session_cache[session_id], "final_output.mp4")
            cmd = f"ffmpeg -y -v warning -i {output_video} -i {audio_path} -c:v copy -c:a aac -strict experimental {final_output}"
            subprocess.run(cmd, shell=True, check=True)

            logger.info(f"Video generation completed for session: {session_id}")
            yield CogPath(final_output)

        except Exception as e:
            logger.error(f"Error during video generation: {str(e)}")
            yield json.dumps({"status": "error", "message": str(e)})
            raise

    def load_preprocessed_data(self, session_path: str):
        """
        Load preprocessed data from disk.

        Args:
            session_path (str): Path to the session data on disk.

        Returns:
            dict: Loaded preprocessed data.

        Raises:
            Exception: If there's an error loading the preprocessed data.
        """
        logger.info(f"Loading preprocessed data from: {session_path}")
        try:
            return torch.load(os.path.join(session_path, "preprocessed_data.pt"))
        except Exception as e:
            logger.error(f"Error loading preprocessed data: {str(e)}")
            raise
