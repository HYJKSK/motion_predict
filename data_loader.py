import os
import sys
import random
import torch
import numpy as np
import BVH

from animation_data import AnimationData
from load_skeleton import Skel


def single_to_batch(data):
    for key, value in data.items():
        data[key] = value.unsqueeze(0)
    return data

def process_single_bvh(filename, bodytype, norm_data_dir=None, downsample=4, skel=None, to_batch=False): #$$
    def to_tensor(x):
        return torch.tensor(x).float()
    anim = AnimationData.from_BVH(filename, bodytype=bodytype, downsample=downsample, skel=skel, trim_scale=4 )
    foot_contact = anim.get_foot_contact(transpose=True)  # [4, T]
    content = to_tensor(anim.get_content_input())

    data = {"contentraw": content }

    if to_batch:
        data = single_to_batch(data)

    return data

def save_bvh_from_network_output(nrot, output_path, bodytype=0):
    anim = AnimationData.from_network_output(nrot, bodytype=bodytype)
    bvh, names, ftime = anim.get_BVH()
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    BVH.save(output_path, bvh, names, ftime)

def content_and_phase(filename, bodytype, downsample=4, skel=None):
    anim = AnimationData.from_BVH(filename, bodytype=bodytype, downsample=downsample, skel=skel)
    content = anim.get_content_phase()
    return content