#select best on current iteration, current split
from os import listdir
import json
import numpy as np
import h5py
import sys
import collections
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd


from tqdm import tqdm, trange
import time

import random

sys.path.insert(0, '/workspace/exp-notgan-pretrain-5iter/evaluation')
from generate_summary import generate_summary
from evaluation_metrics import evaluate_summary

from generate_single_summary import generate_single_summary

#e.g. '../../../gluster/exp-notgan-pretrain-5iter/reg0.7/exp0'
base_path = sys.argv[1]
dataset = sys.argv[2]
iter = sys.argv[3]
split = sys.argv[4]
method = sys.argv[5]
n_epochs = sys.argv[6]
n_epochs = int(n_epochs)

datasetpath = "../data/" + dataset + "/eccv16_dataset_" + dataset.lower() + "_google_pool5.h5"

def normalization(array):
    a = max(array)
    b = min(array)
    out = (array-b)/(a-b)
    return out
def normalization_1(array):
    a = max(array[1:])
    b = min(array[1:])
    out = (array-b)/(a-b)
    out[:1] = np.NaN
    return out
def normalization_2(array):
    a = max(array[2:])
    b = min(array[2:])
    out = (array-b)/(a-b)
    out[:2] = np.NaN
    return out

#read csv file
print("read csv file ..")
csv_file_path = base_path + f"/iter{iter}/{dataset}/logs/split{split}/scalars.csv"
df = pd.read_csv(csv_file_path,
                header=0,
                usecols=["sparsity_loss_epoch_test",
                "recon_loss_epoch_unsupervised"])
sparsity_score = df["sparsity_loss_epoch_test"].to_numpy()[1:n_epochs+1]
recon_score = df["recon_loss_epoch_unsupervised"].to_numpy()[1:n_epochs+1]

#[100] with first 1 as nan
n_sparsity_score = normalization_1(sparsity_score)

#[100]
n_recon_unsupervised_score = normalization(recon_score)

#[100] with first epoch nan
n_final = - 0.5*n_sparsity_score + 0.5*n_recon_unsupervised_score

#final epoch
chosen_epoch = np.argmax(n_final[5:]) + 5
print("split"+str(split), "epoch " + str(chosen_epoch))

seedpath = base_path + f"/iter{iter}/{dataset}/seed/split{split}"
with open(seedpath + f"/{dataset}_unsupervised_epoch.txt", 'w') as f:
    f.write('{}'.format(chosen_epoch))



all_scores = []
result_path = base_path + f"/iter{iter}/{dataset}/results/split{split}"
with open(result_path+'/'+dataset + "_" + str(chosen_epoch) + ".json") as f:
    data = json.loads(f.read())
    keys = list(data.keys())
for video_name in keys:
    scores = np.asarray(data[video_name])
    all_scores.append(scores)

all_user_summary, all_shot_bound, all_nframes, all_positions = [], [], [], []
with h5py.File(datasetpath, 'r') as hdf:        
    for video_name in keys:
        video_index = video_name[6:]
        
        user_summary = np.array( hdf.get('video_'+video_index+'/user_summary') )
        sb = np.array( hdf.get('video_'+video_index+'/change_points') )
        n_frames = np.array( hdf.get('video_'+video_index+'/n_frames') )
        positions = np.array( hdf.get('video_'+video_index+'/picks') )

        all_user_summary.append(user_summary)
        all_shot_bound.append(sb)
        all_nframes.append(n_frames)
        all_positions.append(positions)

all_summaries = generate_summary(all_shot_bound, all_scores, all_nframes, all_positions)
    
all_f_scores = []

# compare the resulting summary with the ground truth one, for each video
for video_index in range(len(all_summaries)):
    summary = all_summaries[video_index]
    user_summary = all_user_summary[video_index]
    f_score = evaluate_summary(summary, user_summary, method)
    

    all_f_scores.append(f_score)
    

picked_f = np.mean(all_f_scores)

print(f"split{split}: epoch {chosen_epoch}, f: {picked_f}, mean f: {picked_f}")

with open(seedpath + f"/{dataset}_unsupervised_f.json", 'w') as f:
    json.dump(picked_f, f)


