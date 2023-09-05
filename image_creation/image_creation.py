import asyncio
import base64
import json
import logging
import os
import re
import sys
from datetime import datetime
from io import BytesIO
from typing import Any, Callable

import requests
import torch
from diffusers import DiffusionPipeline
from PIL import Image

from ..base import BaseTool

class ImageGenerator():
    ''' 
        Select from local/online SDXL and generate images using the selected model
        Will include support for LoRA models
    '''
    def __init__(self, **kwargs):
        use_local = kwargs.get("use_local", False)
        use_refiner = kwargs.get("use_refiner", False)
        if use_refiner and not use_local:
            logging.warning("Refiners cannot be used when use_local is False")
        if use_local:
            try:
                import torch
                cuda_available = torch.cuda.is_available()
                if not cuda_available:
                    raise Exception("cuda unavailable") 
                free_mem = torch.cuda.mem_get_info()[1] / (1024 ** 3)
                if free_mem < 7.5:
                    logging.warning(f"Not enough memory for base model, default to online models. Require 7.5GB, available: {free_mem:.1f}G")
                    use_local = False
                elif free_mem < 16.5:
                    logging.warning(f"Not enough memory for base model, default to base model only or online model. Require 16.5GB, available: {free_mem:.1f}G")
                    use_refiner = False
            except:
                logging.warning("Torch or cuda cannot be used, default to online models")
                use_local = False
        use_refiner = use_local and use_refiner
        
        self.generate: Callable = None

        if use_local:
            self.base = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", 
                torch_dtype=torch.float16, 
                variant="fp16", 
                use_safetensors=True,
                cache_dir="/workspace/cache/hub/"
            )
            self.base.to("cuda")
            self.generate = self.generate_img_local_base
            print("using local base model")
        else:
            print("using online model")
            self.generate = self.generate_img_online
        if use_refiner:
            self.refiner = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                text_encoder_2=self.base.text_encoder_2,
                vae=self.base.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
                cache_dir="/workspace/cache/hub/"
            )
            self.refiner.to("cuda")
            self.generate = self.generate_img_local_with_refiner
            print("using local base model + refiner")

    def generate_img_online(self, **kwargs):
        '''Generate an image based on the prompt, using octoai SDXL'''
        prompt = kwargs.get("prompt")
        user_id = kwargs.get("user_id")
        negative_prompt = kwargs.get("negative_prompt", "")
        num_images = kwargs.get("num_images", 1)
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


    def generate_img_local_base(self, **kwargs):
        '''Generate an image based on the prompt, using SDXL base model only'''
        prompt = kwargs.get("prompt")
        user_id = kwargs.get("user_id")
        negative_prompt = kwargs.get("negative_prompt", "")
        num_images = kwargs.get("num_images", 1)
        image = self.base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
        ).images[0]

        time_stamp = datetime.now().isoformat(timespec="seconds").replace(":", "_")
        user_dir = f"generated_images/{user_id}/{time_stamp}"
        os.makedirs(user_dir, exist_ok=True)
        image.save(f"{user_dir}/{time_stamp}_0.png")

    def generate_img_local_with_refiner(self, **kwargs):
        '''Generate an image based on the prompt, using SDXL base model + refiner'''
        prompt = kwargs.get("prompt")
        user_id = kwargs.get("user_id")
        negative_prompt = kwargs.get("negative_prompt", "")
        num_images = kwargs.get("num_images", 1)
        image = self.base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            denoising_end=0.8,
            output_type="latent",
        ).images
        image = self.refiner(
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
        image.save(f"{user_dir}/{time_stamp}_0.png")

class ImageCreation(BaseTool):
    name: str = "image_creation"
    description: str = "Tool for generating images using SDXL."
    user_description: str = "You can enable this to generate images using SDXL. To make the tool useful, you need to modify the AI description to have the bot generate response (response, followed by JSON) properly. Your AI description and settings will be overwritten. Save your own settings if you need them."
    usable_by_bot: bool = False

    def __init__(self, func: Callable=None, **kwargs) -> None:
        # All the handlers must have been correctly setup, otherwise Memory is no use, 
        # so if there is any error, we must raise
        kwargs["use_local"] = False
        kwargs["use_refiner"] = False
        self.image_generator = ImageGenerator(**kwargs)
        self.settings_overwrite = {
            "assistantsName" :"Diffy",
            "gptMode" : "Smart",
            "relationship" : "Friend",
            "aiDescription" : [
                "You help me come up with words and phrases that best describe a picture I want to draw. These words and phrases are referred to as prompts. The prompts should be concise and accurate and reflect my needs",
                "You need to converse with me to ask for clarifications and give suggestions",
                "Reply in the following format: \"\"\"your suggestions, questions~~{\"basic_prompt\": general description, \"positive_prompt\": Must haves, \"negative_prompt\": Must not haves}\"\"\"",
                "You don't need to generate the image. Only respond to the user and reply a JSON object following the format."],
            "aiSarcasm": 1.0,
        }
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
        self.image_generator.generate(**{"prompt": prompt,
                        "user_id": user_id, 
                        "negative_prompt": "", 
                        "num_images": 1})
    
    def OnResponseEnd(self, **kwargs):
        asyncio.create_task(self.try_generate_image(**kwargs))
    
    def on_enable(self, *args: Any, **kwargs: Any) -> Any:
        user_id = kwargs.get("user_id")
        current_user_settings = kwargs.get("current_user_settings")
        default_user_settings = kwargs.get("default_user_settings")
        update_user_settings_fn = kwargs.get("update_user_settings_fn")
        for k, v in self.settings_overwrite.items():
            current_user_settings[k] = v
        update_user_settings_fn(user_id, current_user_settings)
    
    def on_disable(self, *args: Any, **kwargs: Any) -> Any:
        user_id = kwargs.get("user_id")
        current_user_settings = kwargs.get("current_user_settings")
        default_user_settings = kwargs.get("default_user_settings")
        update_user_settings_fn = kwargs.get("update_user_settings_fn")
        for k in self.settings_overwrite:
            current_user_settings[k] = default_user_settings[k]
        update_user_settings_fn(user_id, current_user_settings)

    def _run(self, *args, **kwargs):
        pass