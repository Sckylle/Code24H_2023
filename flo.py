from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import requests

import torch
import matplotlib.pyplot as plt
import itertools
import numpy as np

from diffusers import StableDiffusionInpaintPipeline



def negatif(image_original, prompts, limit) :
  processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
  model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

  image = image_original.resize((512, 512)) 

  inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")  
  with torch.no_grad():  
      outputs = model(**inputs)  

  masque = torch.sigmoid(outputs[0]).squeeze().numpy() > limit

  masque_array = np.zeros_like(masque)
  true_indices = np.argwhere(masque)

  for idx in true_indices:
      ligne = tuple(slice(max(0, i - 3), i + 1 + 3) for i in idx)
      masque_array[ligne] = True

  visual_masque = (masque_array * 255).astype(np.uint8)
  image_masque = Image.fromarray(masque_array)
  image_masque = image_masque.resize(image.size) 
  return image_masque
  
mask = negatif(img, ["advertisement"], 0.3)
mask




def modif_image(prompts, imageOrigine, mask, isConda = False, nbStep = 20, guidance = 7) :
  if isConda :
    inpainting = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting").to("cuda")
  else :
    inpainting = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting")

  original_image_size = (imageOrigine.height, imageOrigine.width)

  result = inpainting(prompt=prompts, image=imageOrigine,mask_image=mask,
                      num_inference_steps = nbStep, guidance_scale = guidance).images

  res = result[0].resize((original_image_size[1],original_image_size[0]))
  return res

result = modif_image(["delete text", "white", "remove text", "green"], image, mask, True, 20, 7 )
