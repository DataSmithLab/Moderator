# Moderator: Moderating Text-to-Image Diffusion Models through Fine-grained Context-based Policies

## 1-Prerequisite

### 1-1-Install stable diffusion

```
conda activate /home/featurize/work/ModeratorAE/moderator_env
conda env create --prefix /home/featurize/work/ModeratorAE/moderator_env --file moderator.yaml
#git lfs install
apt-get install git-lfs
#git lfs clone https://huggingface.co/runwayml/stable-diffusion-v1-5.git
git lfs clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0.git
mkdir unet_backup
mkdir files
cd files
mkdir models_finetune
cp stable-diffusion-xl-base-1.0/unet/diffusion_pytorch_model.safetensors stable-diffusion-xl-base-1.0/unet_backup/diffusion_pytorch_model.safetensors
#cp stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin unet_backup/unet_original_diffusion_pytorch_model.bin
```

Install stable diffusion-v-1.5 from hugging face.

### 1-2-Install diffuser

```
pip install git+https://github.com/huggingface/diffusers.git
cd ConceptPermission
pip install -U -r requirements.txt
mkdir data
cd ..
```

Install diffuser module from GitHub.

### 1-3-Other dependecies

```
pip install -e git+https://github.com/CompVis/taming-transformers.git#egg=taming_transformers
pip install -e git+https://github.com/openai/CLIP.git#egg=CLIP
export LLMEthicsPatchHome=/home/featurize/work/ModeratorAE#/root/autodl-fs # input your work dir
pip install xformer==0.0.20
pip install flask
pip install matplotlib
pip install scipy
pip install httpx
pip install socksio
```

### 1-4-Install Ollama
```
curl -fsSL https://ollama.com/install.sh | sh
pip install ollama
ollama pull llama3
```

### 1-4-Clone LLMEthicsPatches

```
git clone https://github.com/WhileBug/LLMEthicsPatches.git
cd LLMEthicsPatches
chmod +x ./init.sh
./init.sh
chmod +x ./test.sh
./test.sh

git clone https://github.com/kohya-ss/sd-scripts.git
git checkout sdxl
```

## 2-Quick Start

### 2-1-Run the test

```
flask --app main_backend run
```
This will start a backend on flask on http://127.0.0.1:5000/
It will provide several interfaces:
- model_edit: pass the config json to backend, returned the path of edited unet model
- img_generate: pass the img generate config json to backedn, return the imgs path
You can find example in main_request.py about how to call. The yaml is only for test, you can also directly pass the args using dict/json formats

### 2-2-How to config

You can edit the task.yaml in the main folder.
It provides these params:
- src_content: input the content you want to moderate
- src_name: the name for src_content, just using for filename, not specific meaning
- dst_content: input the content you want to replace the moderated content to
- dst_name: the name for dst_content, just using for filename, not specific meaning
- task_name: the whole task name, used for storing file name
- plot_img_content: a list of prompts for generating example images
- method: "replace" # "block", "replace"; "mosaic" not fully supported yet, because need manual operations