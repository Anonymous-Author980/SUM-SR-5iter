from os import listdir
import json
import numpy as np
import h5py
import sys
from tqdm import tqdm, trange
 
# adding Folder_2 to the system path
sys.path.insert(0, '/workspace/exp-notgan-pretrain-r0/evaluation')
from generate_summary import generate_summary
from evaluation_metrics import evaluate_summary

datapath = "../../../gluster/exp-notgan-pretrain-r0"
#python evaluate_unsupervisedly.py "soccer" "0" "avg"  
dataset = sys.argv[1]
exp = sys.argv[2]

eval_method = sys.argv[3]

path_dataset = "../data/" + dataset + "/eccv16_dataset_" + dataset.lower() + "_google_pool5.h5"

path = datapath + "/exp" + exp + "/" + dataset + "/results/" #path to max results folder
seedpath = datapath + "/exp" + exp + "/" + dataset + "/seed/"


split = [0, 1, 2, 3, 4] #split index

f_splits = [] #store ave f score for each split
mean_f_splits = []

all_f = {} #store f score for each split, each epoch for latter unsupervised evaluation
mean_all_f = {}

for i in tqdm(split):
    path_i = path + "split" + str(i)
    
    results = listdir(path_i)
    

    results = [name for name in results if 'train' not in name]
    results = [name for name in results if 'test' not in name]
    results = [name for name in results if 'scores' not in name]
    results = [name for name in results if '(' not in name]

    
    results.sort(key=lambda video: int(video[6:-5]))
    
    f_score_epochs = []
    mean_f_score_epochs = []

    for j in trange(len(results)):
        epoch = results[j]
        
        all_scores = []
        
        with open(path_i+'/'+epoch) as f:
            data = json.loads(f.read())
            keys = list(data.keys())

        for video_name in keys:
            scores = np.asarray(data[video_name])
            all_scores.append(scores)
        
        all_user_summary, all_shot_bound, all_nframes, all_positions = [], [], [], []
        with h5py.File(path_dataset, 'r') as hdf:        
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
            f_score = evaluate_summary(summary, user_summary, eval_method)
            

            all_f_scores.append(f_score)
            

        f_score_epochs.append(np.mean(all_f_scores))
        
        
    f_splits.append([max(f_score_epochs), int(np.argmax(f_score_epochs)-1)])
    
    all_f[i] = f_score_epochs
    
    
    print(f"Split: {i}, best epoch: {np.argmax(f_score_epochs)-1}, f: {max(f_score_epochs)}")
    
    

f_final = np.mean([x[0] for x in f_splits])



f_splits.append(f_final)


with open(seedpath + "best_result.txt", "w") as f:
    json.dump(f_splits, f)

with open(seedpath + "all_f.json", "w") as f:
    json.dump(all_f, f)




print(f"Final: {f_final}")



    