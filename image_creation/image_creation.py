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
from PIL import Image

from ..base import BaseTool


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
    headers = {"Content-Type": "application/json", "Authorization": "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IjNkMjMzOTQ5In0.eyJzdWIiOiIzMTRhOTBlNC05MDgzLTQwODgtOTc5Yy04NzMyZDA0ZTI0NTYiLCJ0eXBlIjoidXNlckFjY2Vzc1Rva2VuIiwidGVuYW50SWQiOiI3YzFhOWRkNS0wMmIyLTQ4ZmYtOGMzNy1mNDY3Mjc5ZDNiYzciLCJ1c2VySWQiOiJmYmE4YzIyNS1kY2VlLTRmYmItOGQzZC1mYWMxMWUyZGJiNTYiLCJyb2xlcyI6WyJGRVRDSC1ST0xFUy1CWS1BUEkiXSwicGVybWlzc2lvbnMiOlsiRkVUQ0gtUEVSTUlTU0lPTlMtQlktQVBJIl0sImF1ZCI6IjNkMjMzOTQ5LWEyZmItNGFiMC1iN2VjLTQ2ZjYyNTVjNTEwZSIsImlzcyI6Imh0dHBzOi8vaWRlbnRpdHkub2N0b21sLmFpIiwiaWF0IjoxNjg3OTc2MjA2fQ.k9kdyCWTLPR2m8EIMfZXjYImfFKQ48TX0KB7TMiPIa6bD48Prcam48rio6CsAifbs6N3eo-iZubEI0-EYvGMVlzywa1me07MztK2DyOMWnunbtuxj92E6dMxO0RdVqLe6XT-SEF-yrZCnV51QfLIkmFUPPmq5P6X_W8jSoG9eP1lncX98HPE_6aI_5N85x1xGJj9uReVJZq6RTl4L7-UD9zbBzA7hSVJzHFMNxEc4AiknwgX3_xnyRnPlegJVA8rvfQQxBuFuV9_cI1QsgfA6KSZPRQIw5wSP7rGJqXF5MueRzwikfhtk3EC24QPIV3Rm2VolGWDHiSMH_tw4ucZJQ"}
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
        generate_img(**{"prompt": prompt,
                        "user_id": user_id, 
                        "negative_prompt": "", 
                        "num_images": 1})
    
    def OnResponseEnd(self, **kwargs):
        asyncio.create_task(self.try_generate_image(**kwargs))
    
    def _run(self, *args, **kwargs):
        pass