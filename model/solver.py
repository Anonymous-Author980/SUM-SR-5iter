# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import json
from tqdm import tqdm, trange
import time
import numpy as np
import random

from layers import sLSTM, AAE, random_drop
from utils import TensorboardWriter
from generate_single_summary import generate_single_summary




device = 'cuda' if torch.cuda.is_available() else 'cpu'
label10 = torch.tensor([1.0, 0.0]).cuda()


tau = 0.5
class Solver(object):
    def __init__(self, config=None, train_loader=None, test_loader=None):
        """Class that Builds, Trains and Evaluates SUM-NOTGAN model"""
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.test_low_recon = np.inf
        self.test_smallest_epoch = 0
        self.low_recon = np.inf
        self.smallest_epoch = 0
        self.pretrain_low_recon = np.inf
        self.pretrain_epoch = 0

        #save seed
        #seed random process
        seed = int(time.time())
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        seed_save_path = self.config.seed_dir.joinpath(
               f'{self.config.video_type}_seed.txt')
        with open(seed_save_path, 'w') as f:
           f.write('{}'.format(seed))
        
    def build(self):
        # Build Modules
        self.linear_compress = nn.Linear(
            self.config.input_size,
            self.config.hidden_size).cuda()
        self.selector = sLSTM(
            input_size=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers).cuda()
        self.autoencoder = AAE(
            input_size = self.config.input_size,
            hidden_size = self.config.hidden_size,
            num_layers = self.config.num_layers).cuda()
        #Build Modules for pretraining mask token
        self.mask_embedding = nn.Embedding(
            num_embeddings = 2,
            embedding_dim = self.config.input_size,
            padding_idx = 1).to(device) #1 means non mask, 0 means mask
         #initialize mask embedding from zeros
        with torch.no_grad():
            self.mask_embedding.weight[0] = torch.zeros(self.config.input_size)
        
        self.model = nn.ModuleList([
            self.linear_compress, self.selector, self.autoencoder, self.mask_embedding])

        if self.config.mode == 'train':
            # Build Optimizers
            if self.config.full_mode == 'all':
                '''
                train whole model during training
                '''
                self.model_optimizer = optim.Adam(
                    list(self.selector.parameters())
                    +list(self.linear_compress.parameters())
                    +list(self.autoencoder.parameters()),
                    lr = self.config.lr
                )
            else:
                '''
                do not train reconstructor during training
                '''
                self.model_optimizer = optim.Adam(
                    list(self.selector.parameters())
                    +list(self.linear_compress.parameters()),
                    lr = self.config.lr
                )
            
            #optimizer for pretraining
            self.recon_optimizer = optim.Adam(
                list(self.autoencoder.parameters())
                +list(self.mask_embedding.parameters()),
                lr = self.config.lr
            )
        
        self.writer = TensorboardWriter(str(self.config.log_dir))

        #load model from previous iteration
        if self.config.iter > 0:
            
            
            checkpoint_epoch_file = self.config.prev_dir.joinpath('seed/split' + str(self.config.split_index),
                f'{self.config.video_type}_unsupervised_epoch.txt')
            with open(checkpoint_epoch_file, 'r') as f:
                a = f.readlines()
            checkpoint_epoch = a[0] #string


            checkpoint_model_file = self.config.prev_dir.joinpath('models/split'+str(self.config.split_index), f'epoch-{checkpoint_epoch}.pkl')
            self.model.load_state_dict(torch.load(checkpoint_model_file))
            print("loaded previous model")

    
    reconstruction_loss = nn.MSELoss()


    def sparsity_loss(self, scores):
        """Summary-Length Regularization"""

        return torch.abs(torch.mean(scores) - self.config.summary_rate)
    
    
    def train_recon(self):
        '''
        Train reconstructor for 100 epochs, every epoch randomly drop 75% of the original features (*0) as input
        keep the one with the smallest training reconstruction loss.
        '''
        step = 0

        for epoch_i in trange(self.config.recon_n_epochs, desc='Epoch', ncols=80):
            pretrain_recon_loss_history = []

            for batch_i, coupled_features in enumerate(tqdm(
                    self.train_loader, desc='Batch', ncols=80, leave=False)):
                self.model.train()
                # [batch_size=1, seq_len, 1024]
                image_features = coupled_features[0]
                video_name = coupled_features[1][0]
                # [seq_len, 1024]
                image_features = image_features.view(-1, self.config.input_size)
                # [seq_len, 1024]
                image_features_ = Variable(image_features).cuda()

                #---- Train autoencoder ----#
                if self.config.verbose:
                    tqdm.write('\nTraining autoencoder...')
                
                #[seq_len, 1024] with seq_len*0.75 features be zero
                random_summary, mask_idx, _ = random_drop(image_features_, 0.15, device)
                mask = self.mask_embedding(mask_idx.to(device))
                #[seq_len, 1024] with seq_len*0.75 features be mask (embedding[0])and others be original features
                final_random_summary = random_summary + mask

                reconstructed_features = self.autoencoder(final_random_summary.unsqueeze(1).detach())
                recon_loss = self.reconstruction_loss(image_features_.detach(), reconstructed_features.squeeze(1))

                self.recon_optimizer.zero_grad()
                recon_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.recon_optimizer.step()

                pretrain_recon_loss_history.append(recon_loss.data)
                if self.config.verbose:
                    tqdm.write('Plotting...')
                self.writer.update_loss(recon_loss.data, step, 'pretrain_reconstruction_loss')
            
            reconstruction_loss = torch.stack(pretrain_recon_loss_history).mean()
            if self.config.verbose:
                    tqdm.write('Plotting...')
            self.writer.update_loss(reconstruction_loss, epoch_i, 'pretrain_reconstruction_loss_epoch')

            # Save parameters at checkpoint
            ckpt_path = str(self.config.pretrain_save_dir) + f'/epoch-{epoch_i}.pkl'
            tqdm.write(f'Save parameters at {ckpt_path}')
            torch.save(self.model.state_dict(), ckpt_path)

            self.evaluate_pretrain(epoch_i)

            #save epoch
        pretrain_epoch_save_path = self.config.seed_dir.joinpath(
            f'{self.config.video_type}_pretrain_epoch.txt')
        with open(pretrain_epoch_save_path, 'w') as f:
            f.write('{}'.format(self.pretrain_epoch))



    def evaluate_pretrain(self, epoch_i):
        '''
        evaluate step on pretrain stage
        '''
        self.model.eval()

        #record loss on test set
        reconstruction_loss_history = []

        for video_tensor, video_name, _, _, _ in tqdm(
                self.test_loader, desc='Evaluate', ncols=80, leave=False):

            # [seq_len, 1024]
            video_tensor = video_tensor.view(-1, self.config.input_size)
            video_feature = Variable(video_tensor).cuda()

            #[seq_len, 1024] with seq_len*0.75 features be zero
            random_summary, mask_idx, _ = random_drop(video_feature, 0.15, device)
            mask = self.mask_embedding(mask_idx.to(device))
            #[seq_len, 1024] with seq_len*0.75 features be mask (embedding[0])and others be original features
            final_random_summary = random_summary + mask

            reconstructed_features = self.autoencoder(final_random_summary.unsqueeze(1).detach())
            recon_loss = self.reconstruction_loss(video_feature.detach(), reconstructed_features.squeeze(1))
            reconstruction_loss_history.append(recon_loss.data)
        reconstruction_loss = torch.stack(reconstruction_loss_history).mean()
        if self.config.verbose:
                    tqdm.write('Plotting...')
        self.writer.update_loss(reconstruction_loss, epoch_i, 'pretrain_test_reconstruction_loss_epoch')
        if epoch_i > 0:
            if reconstruction_loss < self.pretrain_low_recon:
                self.pretrain_low_recon = reconstruction_loss
                self.pretrain_epoch = epoch_i





#######################################################################################################################







    
    def train(self):
        step = 0
        #load model from pretrain as the initial
        checkpoint_epoch_file = self.config.seed_dir.joinpath(
            f'{self.config.video_type}_pretrain_epoch.txt')
        with open(checkpoint_epoch_file, 'r') as f:
            a = f.readlines()
        checkpoint_epoch = a[0] #string

        checkpoint_model_file = str(self.config.pretrain_save_dir) + f'/epoch-{checkpoint_epoch}.pkl'
        self.model.load_state_dict(torch.load(checkpoint_model_file))
        
        for epoch_i in trange(self.config.n_epochs, desc='Epoch', ncols=80):
            
            recon_loss_history = []
            spar_loss_history = []
            total_loss_history = []
            
            for batch_i, coupled_features in enumerate(tqdm(
                    self.train_loader, desc='Batch', ncols=80, leave=False)):
                
                self.model.train()

                # [batch_size=1, seq_len, 1024]
                image_features = coupled_features[0]
                video_name = coupled_features[1][0]
                # [seq_len, 1024]
                image_features = image_features.view(-1, self.config.input_size)
                # [seq_len, 1024]
                image_features_ = Variable(image_features).cuda()
                seq_len = image_features_.shape[0]

                #---- Train autoencoder ----#
                if self.config.verbose:
                    tqdm.write('\nTraining whole model...')
                
                # [seq_len, 1, hidden_size]
                original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)
                # [seq_len, 1], [seq_len, 1, hidden_size]
                scores, summary_features = self.selector(original_features, image_features_.detach().unsqueeze(1), label10, tau=tau)
                mask_logits = torch.zeros(seq_len, dtype=torch.long).to(device)
                mask_tensor = self.mask_embedding(mask_logits)
                mask = mask_tensor.unsqueeze(1)*(1-scores).view(-1, 1, 1)
                final_summary = summary_features + mask
                reconstructed_features = self.autoencoder(final_summary)

                recon_loss = self.reconstruction_loss(image_features_.detach(), reconstructed_features.squeeze(1))
                spar_loss = self.sparsity_loss(scores)
                loss = recon_loss+spar_loss

                self.model_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.model_optimizer.step()

                recon_loss_history.append(recon_loss.data)
                spar_loss_history.append(spar_loss.data)
                total_loss_history.append(loss.data)

                tqdm.write(f'reconstruction: {recon_loss.item():.3f}, sparsity: {spar_loss.item():.3f}')

                if self.config.verbose:
                    tqdm.write('Plotting...')
                self.writer.update_loss(recon_loss.data, step, 'reconstruction_loss')
                self.writer.update_loss(spar_loss.data, step, "sparsity_loss")
                self.writer.update_loss(loss.data, step, "model_loss")

                step += 1
                

            
            reconstruction_loss = torch.stack(recon_loss_history).mean()
            sparsity_loss = torch.stack(spar_loss_history).mean()
            total_loss = torch.stack(total_loss_history).mean()

            if self.config.verbose:
                    tqdm.write('Plotting...')
            self.writer.update_loss(reconstruction_loss, epoch_i, 'reconstruction_loss_epoch')
            self.writer.update_loss(sparsity_loss, epoch_i, "sparsity_loss_epoch")
            self.writer.update_loss(total_loss, epoch_i, "model_loss_epoch")

            
            # Save parameters at checkpoint
            ckpt_path = str(self.config.save_dir) + f'/epoch-{epoch_i}.pkl'
            tqdm.write(f'Save parameters at {ckpt_path}')
            torch.save(self.model.state_dict(), ckpt_path)
            
            # auto_save_path = self.config.score_dir.joinpath(
            #     f'{self.config.video_type}_{epoch_i}_train.json')
            # with open(auto_save_path, 'w') as f:
            #     tqdm.write(f'Saving elementwise autoencoder loss at {str(auto_save_path)}.')
            #     json.dump(autoencoder_loss_dict, f)
            # auto_save_path.chmod(0o777)

            self.evaluate(epoch_i, tau)
            self.unsupervised_evaluate(epoch_i, tau)
        #save epoch
        epoch_save_path = self.config.seed_dir.joinpath(
               f'{self.config.video_type}_epoch.txt')
        with open(epoch_save_path, 'w') as f:
           f.write('{}'.format(self.smallest_epoch))

        #save epoch based on test loss
        test_epoch_save_path = self.config.seed_dir.joinpath(
               f'{self.config.video_type}_test_epoch.txt')
        with open(test_epoch_save_path, 'w') as f:
           f.write('{}'.format(self.test_smallest_epoch))

    def evaluate(self, epoch_i, tau):

        self.model.eval()

        out_dict = {}
        
        
        #auto_dict = {}

        #record loss on test set
        reconstruction_loss_history = []
        sparsity_loss_history = []
        total_loss_history = []

        for video_tensor, video_name, _, _, _ in tqdm(
                self.test_loader, desc='Evaluate', ncols=80, leave=False):

            # [seq_len, 1024]
            video_tensor = video_tensor.view(-1, self.config.input_size)
            video_feature = Variable(video_tensor).cuda()
            seq_len = video_feature.shape[0]

            # [seq_len, 1, hidden_size]
            compressed_feature = self.linear_compress(video_feature.detach()).unsqueeze(1)
            
            # [seq_len]
            with torch.no_grad():
                scores, summary_features = self.selector(compressed_feature, video_feature.detach().unsqueeze(1), label10, tau=tau)
                mask_logits = torch.zeros(seq_len, dtype=torch.long).to(device)
                mask_tensor = self.mask_embedding(mask_logits)
                mask = mask_tensor.unsqueeze(1)*(1-scores).view(-1, 1, 1)
                final_summary = summary_features + mask
                reconstructed_features = self.autoencoder(final_summary)
                recon_loss = self.reconstruction_loss(video_feature.detach(), reconstructed_features.squeeze(1))
                spar_loss = self.sparsity_loss(scores)
                loss = recon_loss+spar_loss
                

                reconstruction_loss_history.append(recon_loss.data)
                sparsity_loss_history.append(spar_loss.data)
                total_loss_history.append(loss.data)


                scores = scores.squeeze(1)
                scores = scores.cpu().numpy().tolist() 
               
                out_dict[video_name] = scores
                
                # auto_loss = []
                # for i in range(compressed_feature.size()[0]):
                #     l = self.reconstruction_loss(video_feature.detach()[i], reconstruct_features.squeeze(1)[i])
                #     auto_loss.append(l)
                # auto_loss = torch.stack(auto_loss)
                # auto_loss = auto_loss.cpu().numpy().tolist() 
                # auto_dict[video_name] = auto_loss

        reconstruction_loss = torch.stack(reconstruction_loss_history).mean()
        sparsity_loss = torch.stack(sparsity_loss_history).mean()
        total_loss = torch.stack(total_loss_history).mean()

        if self.config.verbose:
                tqdm.write('Plotting...')
        self.writer.update_loss(reconstruction_loss, epoch_i, 'recon_loss_epoch_test')
        self.writer.update_loss(sparsity_loss, epoch_i, "sparsity_loss_epoch_test")
        self.writer.update_loss(total_loss, epoch_i, "total_loss_epoch_test")
    
        score_save_path = self.config.score_dir.joinpath(
            f'{self.config.video_type}_{epoch_i}.json')
        
        # auto_save_path = self.config.score_dir.joinpath(
        #     f'{self.config.video_type}_{epoch_i}_test.json')
        with open(score_save_path, 'w') as f:
            tqdm.write(f'Saving score at {str(score_save_path)}.')
            json.dump(out_dict, f)
        # with open(auto_save_path, 'w') as f:
        #     tqdm.write(f'Saving element wise auto loss at {str(score_save_path)}.')
        #     json.dump(auto_dict, f)
        score_save_path.chmod(0o777)
        
        if reconstruction_loss < self.test_low_recon:
            self.test_low_recon = reconstruction_loss
            self.test_smallest_epoch = epoch_i
    


    def unsupervised_evaluate(self, epoch_i, tau):
        self.model.eval()
        score_save_path = self.config.score_dir.joinpath(
            f'{self.config.video_type}_{epoch_i}.json')
        with open(score_save_path, 'r') as f:
            scores = json.loads(f.read())
        recon_loss_history = []

        for video_tensor, video_name, video_picks, shot_boundary, n_frames in tqdm(
                self.test_loader, desc='Evaluate', ncols=80, leave=False):
            video_tensor_ = Variable(video_tensor).cuda()
            video_scores = scores[video_name]
            _, featured_summary = generate_single_summary(shot_boundary, video_scores, n_frames, video_picks)

            #[seq_len]
            summary_idx = torch.LongTensor(featured_summary).cuda()
            mask = self.mask_embedding(summary_idx)

            #[seq_len]
            featured_summary = torch.Tensor(featured_summary).cuda()

            #[seq_len, input_size]
            summary = video_tensor_ * featured_summary.unsqueeze(1)
            final_summary = summary + mask

            reconstructed_features = self.autoencoder(final_summary.unsqueeze(1))

            recon_loss = self.reconstruction_loss(video_tensor_, reconstructed_features.squeeze(1))

            recon_loss_history.append(recon_loss.data)
        
        recon_loss = torch.stack(recon_loss_history).mean()
        self.writer.update_loss(recon_loss, epoch_i, 'recon_loss_epoch_unsupervised')
        if recon_loss < self.low_recon:
            self.low_recon = recon_loss
            self.smallest_epoch = epoch_i

if __name__ == '__main__':
    pass

                


    
            


