# Prediction interface for Cog ⚙️
# https://cog.run/python
import time
import os
import sys
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from subprocess import Popen, PIPE, STDOUT
import subprocess
import glob
from argparse import Namespace

from cog import BasePredictor, Input, Path as CogPath
from huggingface_hub import snapshot_download, hf_hub_download

sys.path.insert(0, os.path.abspath("scripts"))


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
        # self.model = torch.load("./weights.pth")
        # Running the downloads in parallel
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
       

    def predict(
        self,
        audio_input: CogPath = Input(
            description="Upload your audio file here.",
            default=None,
        ),
        video_input: CogPath = Input(
            description="Upload your video file here.",
            default=None,
        ),
        bbox_shift: int = Input(
            description="",
            default=0,
        ),
        fps: int = Input(
            description="",
            default=25,
        ),

    ) -> CogPath:
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)

        import main as m
        print("RUN: predict")
        args = Namespace(
            video_path=str(video_input),
            audio_path=str(audio_input),
            bbox_shift=bbox_shift,
            result_dir="/src/results",
            fps=fps,
            batch_size=8,
            output_vid_name=None,
            use_saved_coord=False,
            use_float16=False,
        )
        video_path =  m.main(args)
        return CogPath(video_path)
