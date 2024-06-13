from lib.task_vector import TaskVector
import numpy as np
import torch
import argparse
import json
parser = argparse.ArgumentParser() 
parser.add_argument("--A_path", type=str, default=None, help="Task vector A's path")
parser.add_argument("--B_path", type=str, default=None, help="Task vector B's path")
parser.add_argument("--A_name", type=str, default=None, help="Task vector A's name")
parser.add_argument("--B_name", type=str, default=None, help="Task vector B's name")
parser.add_argument("--record_filename", type=str, default=None, help="record filename for task vectors' similarity")
args = parser.parse_args()

def cosine_similarity(A, B):
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

with open(args.record_filename, "r") as f:
    record_dict = json.load(f)
    f.close()

flag=0
if args.A_name in record_dict:
    if args.B_name in record_dict[args.A_name]:
        flag=1
if args.B_name in record_dict:
    if args.A_name in record_dict[args.A_name]:
        flag=1
if flag==0:    
    task_vector_A=TaskVector(vector_path=args.A_path)
    task_vector_B=TaskVector(vector_path=args.B_path)
    try:
        cosine_val = cosine_similarity(task_vector_A.vector, task_vector_B.vector)
        print(args.A_name, args.B_name, "success")
        with open(args.record_filename, "w+") as f:
            if args.A_name not in record_dict:
                record_dict[args.A_name]={}
            record_dict[args.A_name][args.B_name]=cosine_val
            json.dump(record_dict, f)
    except:
        print(args.A_name, args.B_name, "failed")