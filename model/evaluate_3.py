#use reconstructor picked from unsupervisedly and calculate reconstructor loss for each epoch and pick the lowest one
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

from generate_single_summary import generate_single_summary
#python evaluate_unsupervisedly.py "TVSum"
config = get_config(mode='test')
dataset = config.video_type

datapath = "/../../../gluster/exp-notgan-tvsum-r6/"
seedpath = datapath + dataset  +  "/seed/"
modelpath = datapath + dataset   + "/models/"
resultpath = datapath  + dataset  + "/results/"
datasetpath = "../data/" + dataset + "/eccv16_dataset_" + dataset.lower() + "_google_pool5.h5"

f_score_path = seedpath + "mean_all_f.json"
with open(f_score_path, 'r') as f:
    all_f = json.load(f)



solver = Solver(config)


splits = [0, 1, 2, 3, 4]
recon_all_split = {}
picked_epoch_2 = {}
f_splits = []

for split in splits:
    #get the previous picked epoch
    epoch_file = f"{seedpath}split{str(split)}/{dataset}_test_epoch.txt"
    with open(epoch_file, 'r') as f:
        a = f.readlines()
    epoch = a[0] #string
    
    #get the picked model for reconstructor
    model_file = f"{modelpath}split{str(split)}/epoch-{epoch}.pkl"
    solver.build()
    model = solver.model
    model.load_state_dict(torch.load(model_file))
    model.eval()
    autoencoder = model[2]
    autoencoder.eval()

    #get the data loader
    test_loader = get_loader("test", dataset, split)

    #set hyperparameters
    low_recon = - np.inf
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
        if recon_loss > low_recon:
            low_recon = recon_loss
            smallest_epoch = epoch_i
    
    #save recon loss
    recon_all_split[split] = torch.stack(split_recon).cpu().numpy().tolist()
    
    #save picked epoch
    picked_epoch_2[split] = smallest_epoch

    #picked_f
    picked_f = all_f[str(split)][smallest_epoch+1]
    f_splits.append(picked_f)

    print(f"split{split}: epoch {smallest_epoch}, f: {picked_f}")

a = float(np.mean(f_splits))
print("final f score: ", a)

with open(seedpath + "evaluate_3_recon.json", "w") as f:
    json.dump(recon_all_split, f)

with open(seedpath + "picked_epoch_3.json", "w") as f:
    json.dump(picked_epoch_2, f)

with open(seedpath + "unsupervised_result_3.json", "w") as f:
    json.dump(f_splits, f)






