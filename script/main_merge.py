from lib.task_vector import TaskVector
from lib.utils_merge import merge_methods, state_dict_to_vector, vector_to_state_dict
import torch
import numpy as np

redundant = 'topk20'
elect = 'mass'
agg = 'dis-mean'
scale = 'linear+0.8+2.51+0.1'
merge_func = redundant+"_"+elect+"_"+agg+"_"+scale


tv_filenames=[
    '/root/autodl-fs/LLMEthicsPatches/task_vectors/5-Gambling-1-Casino-1000-1.0.npy',
    '/root/autodl-fs/LLMEthicsPatches/task_vectors/5-Gambling-10-Lottery-1000-1.0.npy',
    '/root/autodl-fs/LLMEthicsPatches/task_vectors/5-Gambling-11-Jackpot-1000-1.0.npy',
    '/root/autodl-fs/LLMEthicsPatches/task_vectors/5-Gambling-12-Bookmaker-1000-1.0.npy',
    '/root/autodl-fs/LLMEthicsPatches/task_vectors/5-Gambling-13-Online_gambling-1000-1.0.npy',
    '/root/autodl-fs/LLMEthicsPatches/task_vectors/5-Gambling-2-Bets-1000-1.0.npy',
    '/root/autodl-fs/LLMEthicsPatches/task_vectors/5-Gambling-3-Playing_Card-1000-1.0.npy',
    '/root/autodl-fs/LLMEthicsPatches/task_vectors/5-Gambling-4-Dice-1000-1.0.npy',
    '/root/autodl-fs/LLMEthicsPatches/task_vectors/5-Gambling-5-Wagering-1000-1.0.npy',
    '/root/autodl-fs/LLMEthicsPatches/task_vectors/5-Gambling-6-Poker-1000-1.0.npy'
]

merged_tv_filename = '/root/autodl-fs/LLMEthicsPatches/task_vectors/gambling_tie_merging.npy'

#if __name__ == "__main__":
