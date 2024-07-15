# Moderator: Moderating Text-to-Image Diffusion Models through Fine-grained Context-based Policies

## 1-Prerequisite

### 1-1-Install environment

```shell
cd Moderator
conda env create --prefix moderator --file moderator.yaml
conda activate moderator
```

Install diffuser module from GitHub.

### 1-2-Initialize the environment

```shell
chmod +x ./init.sh
bash ./init.sh
```

### 1-3-Install Ollama and pull Llama3
```shell
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3
```

## 2-Quick Start

### 2-1-Run the test
First run the command below to start the backend
```shell
python main_backend.py
```
This will start a backend on flask on http://127.0.0.1:7417/
It will provide several interfaces:
- pretrain_img_generate: Pass the prompt to generate images on pretrained models. See example in (AE_policy_result.py)[AE_policy_result.py]
- img_generate: Pass the prompt to generate images on moderated models. See example in (AE_policy_result.py)[AE_policy_result.py]
- craft_config: Pass the config to generate policy. See example in (AE_policy_craft.py)[AE_policy_craft.py]
You can craft scripts to use the interfaces, and you can also use our frontend interface.

Then you can access http://localhost:7417/index to use the frontend interface.