import asyncio
import base64
import json
import os
import re
import sys
from datetime import datetime
from io import BytesIO
from typing import Callable

import requests
import torch
from diffusers import DiffusionPipeline
from PIL import Image

from ..base import BaseTool

base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    variant="fp16", 
    use_safetensors=True,
    cache_dir="/workspace/cache/hub/"
)
base.to("cuda")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    cache_dir="/workspace/cache/hub/"
)
refiner.to("cuda")


def generate_img(prompt, user_id, negative_prompt="", num_images=1):
    '''Generate an image based on the prompt, using SDXL'''
    endpoint_url = "https://stable-diffusion-xl-demo-kk0powt97tmb.octoai.cloud"
    model_name = "stable-diffusion-xl-base-1.0"
    '''Call the model'''
    payload = {
        "prompt" : prompt,
        "negative_prompt" : negative_prompt,
        "steps": 30,
        "num_images": num_images,
        # "style_preset" : "analog-film",
        # "seed": 1234,
        # "lora": {"LowRA": 0.2, "pixelart": 0.2},
    }
    payload_json = json.dumps(payload)
    bearer_key = os.environ["OCTO_AI_KEY"]
    headers = {"Content-Type": "application/json", f"Authorization": "Bearer {bearer_key}"}
    response = requests.post(endpoint_url+"/predict",
    headers=headers, data=payload_json)

    data = response.json()

    time_stamp = datetime.now().isoformat(timespec="seconds").replace(":", "_")
    user_dir = f"generated_images/{user_id}/{time_stamp}"
    os.makedirs(user_dir, exist_ok=True)
    for i in range(num_images):
        img_data = base64.b64decode(data['completion'][f'image_{i}'])

        # Open the image with PIL
        img = Image.open(BytesIO(img_data))
        img.save(f"{user_dir}/{time_stamp}_{i}.png")

def generate_img_local(prompt, user_id, negative_prompt="", num_images=1):
    image = base(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        denoising_end=0.8,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        denoising_start=0.8,
        image=image,
    ).images[0]

    time_stamp = datetime.now().isoformat(timespec="seconds").replace(":", "_")
    user_dir = f"generated_images/{user_id}/{time_stamp}"
    os.makedirs(user_dir, exist_ok=True)
    # image = Image.fromarray(image)
    image.save(f"{user_dir}/{time_stamp}_0.png")


class ImageCreation(BaseTool):
    name: str = "image_creation"
    description: str = "Tool for generating images using SDXL."
    user_description: str = "You can enable this to generate images using SDXL. To make the tool useful, you need to modify the AI description to have the bot generate response (response, followed by JSON) properly."
    usable_by_bot: bool = False

    def __init__(self, func: Callable=None, **kwargs) -> None:
        # All the handlers must have been correctly setup, otherwise Memory is no use, 
        # so if there is any error, we must raise
        OnResponseEnd = kwargs.get("OnResponseEnd")
        OnResponseEnd += self.OnResponseEnd

        super().__init__(None)

    def extract_json(self, msg) -> dict:
        # Find JSON object
        match = re.search(r'\{.*\}', msg, re.DOTALL)
        json_str = match.group()

        # Convert JSON string to Python dictionary
        json_dict = json.loads(json_str)
        return json_dict
    
    async def try_generate_image(self, **kwargs):
        user_id = kwargs.get("user_id")
        message = kwargs.get("message")
        websocket = kwargs.get("websocket")
        assert message["role"] == "assistant"
        msg = message["content"]
        try:
            generation_json = self.extract_json(msg)
        except Exception as e:
            print(f"Failed to extract json from message {msg}.\nError: {e}")
            return
        print(generation_json)
        prompt = ""
        for k in ["prompt", "basic_prompt", "positive_prompt"]:
            if k not in generation_json:
                continue
            prompt += generation_json.pop(k) + ","
        negative = generation_json.pop("negative_prompt")
        keys_left = list(generation_json.keys())
        for k in keys_left:
            prompt += generation_json.pop(k) + ","
        print(f"Prompt: {prompt}")
        await websocket.send_text(f"Generating images. Please wait. You can access your images at: /generated_images/{user_id}/")
        generate_img_local(**{"prompt": prompt,
                        "user_id": user_id, 
                        "negative_prompt": "", 
                        "num_images": 1})
    
    def OnResponseEnd(self, **kwargs):
        asyncio.create_task(self.try_generate_image(**kwargs))
    
    def _run(self, *args, **kwargs):
        pass