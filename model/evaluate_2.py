#use reconstructor picked from unsupervisedly and calculate reconstructor loss for each epoch and pick the lowest one
#picked epoch first then calculate f score
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
from configs import get_config
from solver import Solver
from data_loader import get_loader

from tqdm import tqdm, trange
import time

import random

from layers import sLSTM, AE, VAE, AAE
# adding Folder_2 to the system path
sys.path.insert(0, '/workspace/exp-notgan-pretrain-r0/evaluation')
from generate_summary import generate_summary
from evaluation_metrics import evaluate_summary

from generate_single_summary import generate_single_summary
#python evaluate_unsupervisedly.py "TVSum"
config = get_config(mode='test')
dataset = config.video_type
exp = config.exp

datapath = "/../../../gluster/exp-notgan-pretrain-r0"
seedpath = datapath + '/exp' + str(exp) + '/' + dataset + "/seed/"
modelpath = datapath + '/exp' + str(exp) + '/' + dataset + "/models/"
resultpath = datapath + '/exp' + str(exp) + '/' + dataset + "/results/"
datasetpath = "../data/" + dataset + "/eccv16_dataset_" + dataset.lower() + "_google_pool5.h5"

datapath2 = "/../../../gluster/exp-notgan-pretrain-r0"
seedpath2 = datapath2 + '/exp' + str(exp) + '/' + dataset + "/seed/"

solver = Solver(config)


splits = [0, 1, 2, 3, 4]
recon_all_split = {}
picked_epoch_2 = {}


for split in splits:
    #get the picked epoch in pretrain
    checkpoint_epoch_file = seedpath + f'split{str(split)}/{config.video_type}_pretrain_epoch.txt'
    with open(checkpoint_epoch_file, 'r') as f:
        a = f.readlines()
    checkpoint_epoch = a[0] #string

    #get the picked model for reconstructor
    model_file = f"{modelpath}split{str(split)}/pretrain/epoch-{checkpoint_epoch}.pkl"
    solver.build()
    model = solver.model
    model.load_state_dict(torch.load(model_file))
    model.eval()
    autoencoder = model[2]
    autoencoder.eval()

    #get the data loader
    test_loader = get_loader("test", dataset, split)

    #set hyperparameters
    low_recon = np.inf
    smallest_epoch = 0
    split_recon = []
    for epoch_i in trange(config.n_epochs):
        score_save_path = resultpath + "split" + str(split) + "/" + dataset + "_" + str(epoch_i) + ".json"
        with open(score_save_path, 'r') as f:
            scores = json.loads(f.read())
        recon_loss_history = []

        for video_tensor, video_name, video_picks, shot_boundary, n_frames in tqdm(
                test_loader, desc='Evaluate', ncols=80, leave=False):
            video_tensor_ = Variable(video_tensor).cuda()
            video_scores = scores[video_name]

            _, featured_summary = generate_single_summary(shot_boundary, video_scores, n_frames, video_picks)

            #[seq_len]
            featured_summary = torch.Tensor(featured_summary).cuda()

            #[seq_len, input_size]
            summary = video_tensor_ * featured_summary.unsqueeze(1)

            reconstructed_features = autoencoder(summary.unsqueeze(1))

            recon_loss = solver.reconstruction_loss(video_tensor_, reconstructed_features.squeeze(1))
            recon_loss_history.append(recon_loss.data)
        
        recon_loss = torch.stack(recon_loss_history).mean()
        split_recon.append(recon_loss)
        if recon_loss < low_recon:
            low_recon = recon_loss
            smallest_epoch = epoch_i
    
    #save recon loss
    recon_all_split[split] = torch.stack(split_recon).cpu().numpy().tolist()
    
    #save picked epoch
    picked_epoch_2[split] = smallest_epoch

#calculate f score
all_f = []
all_f_mean = []

for i in tqdm(splits):
    path_i = resultpath + "split" + str(i)
    epoch = picked_epoch_2[i]
    all_scores = []
        
    with open(path_i+'/'+dataset + "_" + str(epoch) + ".json") as f:
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
    all_mean_f_scores = []
    
    # compare the resulting summary with the ground truth one, for each video
    for video_index in range(len(all_summaries)):
        summary = all_summaries[video_index]
        user_summary = all_user_summary[video_index]
        f_score = evaluate_summary(summary, user_summary, 'max')
        mean_f_score = evaluate_summary(summary, user_summary, 'avg')

        all_f_scores.append(f_score)
        all_mean_f_scores.append(mean_f_score)
    
    picked_f = np.mean(all_f_scores)
    picked_mean_f = np.mean(all_mean_f_scores)
    all_f.append(picked_f)
    all_f_mean.append(picked_mean_f)

    print(f"split{i}: epoch {epoch}, f: {picked_f}, mean f: {picked_mean_f}")

a = float(np.mean(all_f))
b = float(np.mean(all_f_mean))
print("final max f score: ", a)
print("final mean f score: ", b)

f_splits = {}
f_splits["max"] = all_f
f_splits["avg"] = all_f_mean
with open(seedpath2 + "evaluate_2_recon.json", "w") as f:
    json.dump(recon_all_split, f)

with open(seedpath2 + "picked_epoch_2.json", "w") as f:
    json.dump(picked_epoch_2, f)

with open(seedpath2 + "unsupervised_result_2.json", "w") as f:
    json.dump(f_splits, f)






