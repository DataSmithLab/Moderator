apt-get install git-lfs

# Init some folders
mkdir data
mkdir files
cd files
mkdir models_finetune
mkdir task_vectors
mkdir models_edited

# Swtich outside
cd ..

# Install stable-diffusion-XL
git lfs clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0.git
cd stable-diffusion-xl-base-1.0
mkdir unet_backup
cp unet/diffusion_pytorch_model.fp16.safetensors unet_backup/diffusion_pytorch_model.fp16.safetensors
cd ..

# Install encoder for sd-scripts
git lfs clone https://huggingface.co/openai/clip-vit-large-patch14
git lfs clone https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k

# Initialize virtual environments
conda env create --prefix moderator_conda --file moderator.yaml
conda activate moderator_conda
pip install -r requirements.txt


# Install ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama pull llama3:8b