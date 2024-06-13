from diffusers import StableDiffusionPipeline
import torch

#model_path = "/root/autodl-fs/LLMEthicsPatches/models_finetune/finetuned_Unet-lora-gun"
pipe = StableDiffusionPipeline.from_pretrained("/root/autodl-fs/stable-diffusion-v1-5-backend", torch_dtype=torch.float16)
#pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

prompt = "gun"
for i in range(20):
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    image.save("lora_test-100/pretrain_gun"+str(i)+".png")


lora_weight = torch.load("/root/autodl-fs/LLMEthicsPatches/models_finetune/finetuned_Unet-lora-gun-100/pytorch_lora_weights.bin")    
neg_lora_weight = {}
for key in lora_weight:
    neg_lora_weight[key]=-lora_weight[key]
torch.save(neg_lora_weight, "/root/autodl-fs/LLMEthicsPatches/models_finetune/finetuned_Unet-lora-gun-100/pytorch_lora_weights.bin")

model_path = "/root/autodl-fs/LLMEthicsPatches/models_finetune/finetuned_Unet-lora-gun"
pipe = StableDiffusionPipeline.from_pretrained("/root/autodl-fs/stable-diffusion-v1-5-backend", torch_dtype=torch.float16)
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

prompt = "gun"
for i in range(20):
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    image.save("lora_test-100/edit_gun"+str(i)+".png")