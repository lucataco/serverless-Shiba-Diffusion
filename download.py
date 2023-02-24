# In this file, we define download_model
# It runs during container build time to get model weights built into the container
import os
import torch
import requests
from tqdm import tqdm
from diffusers import StableDiffusionPipeline

def download_model():
    global model
    # Download weights
    ckpt_url = "https://huggingface.co/lucataco/Shiba/resolve/main/Shiba_Diffusion.ckpt"
    print(f"Downloading {ckpt_url}...")
    resp = requests.get(ckpt_url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open("local.ckpt",'wb') as file, tqdm(
        desc="Downloading",
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)      

    os.system("python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path ./local.ckpt --dump_path sd_weights/")
    model = StableDiffusionPipeline.from_pretrained(
        "sd_weights/", 
        safety_checker=None,
        revision="fp16",
        torch_dtype=torch.float16
    )

if __name__ == "__main__":
    download_model()