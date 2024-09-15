# Prediction interface for Cog ⚙️
# https://cog.run/python
import time
import os
import sys
import shutil
import json
from typing import Optional
from collections import OrderedDict
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from subprocess import Popen, PIPE, STDOUT
import subprocess
import glob
from argparse import Namespace

from cog import BasePredictor, Input, Path as CogPath
from huggingface_hub import snapshot_download, hf_hub_download

sys.path.insert(0, os.path.abspath("scripts"))

# Constants
SESSION_CACHE_DIR = "/tmp/data/session"
MAX_SESSIONS_IN_MEMORY = 1  # Number of sessions to keep in memory

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

def download_hf_model(name, dest):
    cache_dir = "./cache"
    try:
        print(f"[~] Downloading model with:", name)
        print(f"[~] Downloading model to:", dest)
        snapshot_download(
                cache_dir=cache_dir,
                local_dir=dest,
                repo_id=name,
                allow_patterns=["*.json", "*.bin"],
                ignore_patterns=["*.safetensors", "*.msgpack","*.h5", "*.ot", ],
        )
    except Exception as e :
        print(e)

def download_model(url, dest):
    # Check if the destination directory exists, if not, create it
    dest_dir = os.path.dirname(dest)
    os.makedirs(dest_dir, exist_ok=True)

    # Check if the destination file already exists
    if not os.path.exists(dest):
        start = time.time()
        print("Downloading URL: ", url)
        print("Downloading to: ", dest)
        subprocess.check_call(["pget", url, dest], close_fds=False)
        print("Downloading took: ", time.time() - start)
    else:
        print(f"File already exists: {dest}")

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print("RUN: setup")
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(download_model, url, dest) for url, dest in downloads
            ]
            futures += [
                executor.submit(download_hf_model, name, dest) for name, dest in downloads_hf
            ]
            futures += [
                executor.submit(hf_hub_download, cache_dir="./cache", local_dir="models/dwpose", repo_id="yzd-v/DWPose", filename="dw-ll_ucoco_384.pth")
            ]

        # Ensure all downloads are complete before proceeding
        for future in futures:
            future.result()
        
        # Initialize session cache
        self.session_cache = OrderedDict()

    def precompute_video_data(self, session_id: str, video_path: str, bbox_shift: int):
        import main as m
        args = Namespace(
            video_path=video_path,
            audio_path=None,
            bbox_shift=bbox_shift,
            result_dir=os.path.join(SESSION_CACHE_DIR, session_id),
            fps=25,
            batch_size=8,
            output_vid_name=None,
            use_saved_coord=False,
            use_float16=False,
        )
        m.precompute_video_data(args)
        
        # Update in-memory cache
        self.session_cache[session_id] = args.result_dir
        if len(self.session_cache) > MAX_SESSIONS_IN_MEMORY:
            oldest_session = next(iter(self.session_cache))
            del self.session_cache[oldest_session]

    def predict(
        self,
        session_id: str = Input(description="Session ID for caching", default=None),
        audio_input: Optional[CogPath] = Input(description="Upload your audio file here.", default=None),
        video_input: Optional[CogPath] = Input(description="Upload your video file here.", default=None),
        bbox_shift: int = Input(description="", default=0),
        fps: int = Input(description="", default=25),
    ) -> CogPath:
        """Run a single prediction on the model"""
        if session_id is None:
            raise ValueError("session_id is required")

        if video_input and audio_input:
            raise ValueError("Provide either video_input or audio_input, not both")

        if video_input:
            # Case 1: Precompute video data
            self.precompute_video_data(session_id, str(video_input), bbox_shift)
            return CogPath(json.dumps({"status": "success", "message": "Video data precomputed and stored"}))

        elif audio_input:
            # Case 2: Generate lip-synced video
            if session_id not in self.session_cache:
                session_path = os.path.join(SESSION_CACHE_DIR, session_id)
                if not os.path.exists(session_path):
                    raise ValueError(f"No precomputed data found for session {session_id}")
                self.session_cache[session_id] = session_path

            import main as m
            args = Namespace(
                video_path=None,
                audio_path=str(audio_input),
                bbox_shift=bbox_shift,
                result_dir=self.session_cache[session_id],
                fps=fps,
                batch_size=8,
                output_vid_name=None,
                use_saved_coord=True,
                use_float16=False,
            )
            video_path = m.main(args)
            return CogPath(video_path)

        else:
            raise ValueError("Provide either video_input or audio_input")
