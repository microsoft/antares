import torch
import autort
import os, sys

try:
  from diffusers import StableDiffusionUpscalePipeline
  import transformers
  from PIL import Image
  import numpy as np
except:
  raise Exception(f'Application dependencies are missing, please install with: {sys.executable} -m pip install transformers diffusers Pillow numpy')


data_path = autort.download('./samurai_nn.png', 'https://huggingface.co/datasets/ghostplant/data-collections/resolve/main/samurai_nn.png?download=true', is_zip=False)
target_path = 'samurai_nn_diffused.png'

low_res_img = (torch.from_numpy(np.array(Image.open(data_path).resize((80, 80)))) / 255.0).permute(2, 0, 1).unsqueeze(0)

pipeline = StableDiffusionUpscalePipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", revision="main", torch_dtype=torch.float32
).to(autort.device())

def run():
  return pipeline(prompt="a cartoon face", image=low_res_img).images

upscaled_image = run()
print(f'Image converted from `{data_path}` to `{target_path}`..')
upscaled_image[0].save(target_path)
