from diffusers import StableDiffusionPipeline
import torch
import datetime

boobies_finetuned_model_id = "/root/autodl-fs/finetuned_stable_diffusion_1327"
original_model_id = "/root/autodl-fs/stable-diffusion-v1-5"

model_id = original_model_id
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    safety_checker=None,
)
pipe = pipe.to("cuda")

for i in range(20):
    prompt = "naked, woman"
    images = pipe(prompt).images
    for idx, image in enumerate(images): 
        image.save("./imgs_original_0721/original"+str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+".png")