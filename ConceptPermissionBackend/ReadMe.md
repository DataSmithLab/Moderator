# Access Control for Stable Diffusion

## 1-Prerequisite

### 1-1-Install stable diffusion

```
git lfs install
git lfs clone https://huggingface.co/runwayml/stable-diffusion-v1-5.git
```

Install stable diffusion-v-1.5 from hugging face.

### 1-2-Install diffuser

```
pip install git+https://github.com/huggingface/diffusers.git
pip install -U -r requirements.txt
```

Install diffuser module from GitHub.

### 1-3-Other dependecies

```
pip install -e git+https://github.com/CompVis/taming-transformers.git
pip install -e git+https://github.com/openai/CLIP.git
export LLMEthicsPatchHome=/root/autodl-fs # input your work dir
pip install xformer=0.0.20
pip install flask
```

### 1-4-Clone LLMEthicsPatches

```
git clone https://github.com/WhileBug/LLMEthicsPatches.git
cd LLMEthicsPatches
chmod +x ./init.sh
./init.sh
chmod +x ./test.sh
./test.sh
```

## 2-Quick Start

### 2-1-Run the test

```
flask --app main_flask run
```
This will start a backend on flask on http://127.0.0.1:5000/
It will provide several interfaces:
- model_edit: pass the config json to backend, returned the path of edited unet model
- img_generate: pass the img generate config json to backedn, return the imgs path
You can find example in main_request.py about how to call. The yaml is only for test, you can also directly pass the args using dict/json formats

### 2-2-How to config

```
task_vector_applied: 0
merge: False
task_vectors:
- gen_img_num_per_prompt: 10
  input_data_init: 1
  input_num: 100
  name: gun
  operator: '-'
  names: [gun]
  prompts: ["gun"]
  real_prompts: ["gun"]
  scale: 1.0
  train_step: 1000
  trained: 1
  saved: 1
- gen_img_num_per_prompt: 10
  input_data_init: 1
  input_num: 100
  name: DollGun
  operator: '+'
  names: [DollGun]
  prompts: ["gun"]
  real_prompts: ["Doraemon doll"]
  scale: 1.0
  train_step: 1000
  trained: 1
  saved: 1
```
How to run a test?
The above is the demo in config_edit.yaml, you can modify it. This is the config for editing
For each task vector:
- input_data_init: 1 means the data for training is ready, 0 means not ready, and will trigger data generation
- input_num: this indicates how many images are used to finetune a model on a given task vector
- name: the name of the task vectors
- operator: '-' means neg the task vector, '+' means add the task vector
- names: [gun], name for each prompt, is used for img name in os
- prompts: ["gun"], caption for the finetuning
- real_prompts: ["gun"], generated prompt for image generate, it can be different from prompts
- scale: scale for each task vector
- train_step: training steps for each task vector, recommend for 1000
- trained: 1 means the model already finetuned
- saved: 1 means the task vector already and saved
Multiple task vectors configuration will accumulate.
Other config options:
- task_vector_applied: 1 indicates the task vectors already applied to the model
- merge: False indicates using the accumulated task vectors, True indicates merge(no simply add)
P.S: It will take a long time to finetune (about 1000-2000 seconds per task vector) and img generate (about 5 seconds per image, and 15 seconds to init the model), so do not set the response time to very low value

```
folder_name: /exp-5/gun2toy-after-test
img_gen_num: 20
img_names: [gunPoint]
img_prompts: ["A pretty girl with long blond hair, staring at me, with a gun pointed at her head"]
gen_img_num_per_prompt: 1
unet_path: "/root/autodl-fs/LLMEthicsPatches/models_edited/unet_'-gun_1.0''+MosaicPureGun_1.0'.bin"
```
The above is the demo in config_img.yaml, you can modify it. This is the config for img generating
Config options:
- folder_name: saved data folder name, must be subfolder name in /data
- img_gen_num: generated img number for each prompt
- img_names: generated img names
- img_prompts: generated img prompts
- gen_img_num_per_prompt: 1 # default to 1, for efficiency and memory consumption
- unet_path: the unet path