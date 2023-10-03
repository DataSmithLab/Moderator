import torch
from transformers import CLIPProcessor, CLIPModel
from scipy.optimize import lsq_linear
import numpy as np
from lib.task_vector import TaskVector
import numpy as np
import torch
import argparse
import json

# 加载CLIP模型和处理器
model = CLIPModel.from_pretrained("/root/autodl-fs/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("/root/autodl-fs/clip-vit-large-patch14")

def linear_solve(prompt1: str, prompt2: str, prompt3: str) -> (float, float):
    # 获取prompt的embedding
    prompt1_embedding = model.get_text_features(processor(prompt1, return_tensors="pt")["input_ids"]).detach().numpy().flatten()
    prompt2_embedding = model.get_text_features(processor(prompt2, return_tensors="pt")["input_ids"]).detach().numpy().flatten()
    prompt3_embedding = model.get_text_features(processor(prompt3, return_tensors="pt")["input_ids"]).detach().numpy().flatten()

    # 构造线性方程组
    A = np.vstack([prompt2_embedding, prompt3_embedding]).T
    b = prompt1_embedding.T

    # 使用最小二乘法求解线性方程组
    res = lsq_linear(A, b)
    
    return res.x[0], res.x[1]

def linear_solve_validate(prompt1: str, prompt2: str, prompt3: str, a:float, b:float) -> (float):
    # 获取prompt的embedding
    prompt1_embedding = model.get_text_features(processor(prompt1, return_tensors="pt")["input_ids"]).detach().numpy()
    prompt2_embedding = model.get_text_features(processor(prompt2, return_tensors="pt")["input_ids"]).detach().numpy()
    prompt3_embedding = model.get_text_features(processor(prompt3, return_tensors="pt")["input_ids"]).detach().numpy()

    # 计算两个prompt embedding之间的余弦相似度
    similarity = torch.nn.functional.cosine_similarity(torch.tensor(prompt1_embedding), torch.tensor(prompt2_embedding*a+prompt3_embedding*b)).item()
    
    return similarity

def task_vector_cosine_similarity(A, B):
    dot_product = 0
    norm_A = 0
    norm_B = 0
    for key in A.keys():
        if 'weight' in key:
            a=A[key]
            b=B[key]
            a = torch.flatten(a)
            b = torch.flatten(b)
            dot_product += torch.dot(a, b)
            norm_A += torch.norm(a) ** 2
            norm_B += torch.norm(b) ** 2
        else:
            pass
    similarity = dot_product / (torch.sqrt(norm_A) * torch.sqrt(norm_B))
    #print(similarity)
    return float(similarity)

# 示例
prompt1 = "sex doll"
prompt2 = "boobies"
prompt3 = "nudity"

a, b = linear_solve(prompt1, prompt2, prompt3)
print(f"a: {a}, b: {b}")
#a=0.39049267768859863
#b=0.4189615845680237

similarity = linear_solve_validate(prompt1, prompt2, prompt3, a, b)
print(f"text embedding similarity:{similarity}")

task_vector_1=TaskVector(vector_path="/root/autodl-fs/LLMEthicsPatches/task_vectors/2-Adult_Novelty-4-Sex_doll-1000-1.0.npy")
task_vector_2=TaskVector(vector_path="/root/autodl-fs/LLMEthicsPatches/task_vectors/only_boobies-1000-1.0.npy")
task_vector_3=TaskVector(vector_path="/root/autodl-fs/LLMEthicsPatches/task_vectors/6-Nudity-1-Nudity-1000-1.0.npy")
print(type(task_vector_2.vector.keys())) 
new_vector_2 = task_vector_2*a
#print("2 finish")
#print(task_vector_3.vector)
new_vector_3 = task_vector_3*b
print("3 finish")
new_vector = new_vector_2+new_vector_3
cosine_val = task_vector_cosine_similarity(task_vector_1.vector, new_vector.vector)
print(f"task vector similarity:{cosine_val}")