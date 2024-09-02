# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class sLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        """Scoring LSTM"""
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.linear = nn.Sequential(
            nn.Linear(hidden_size * 2, 2),
            )  # bidirection => scalar
        self.out = nn.Softmax(dim = -1)
    def forward(self, features, original_features, label10, tau = 1, init_hidden=None):
        """
        Args:
            features: [seq_len, 1, 512] (compressed pool5 features)
            original_features: [seq_len, 1, 1024] (original pool5 features)
            label10: [2] a tensor looks like [1, 0] indicates the sample for gumble softmax
            tau: temperature for gumbel softmax, default is 1, will decrease along training
            hard: boolean. to decide weather we use one hot vector or soft gumble for sampling
        Return:
            scores: [seq_len, 1]
            weighted_features: [seq_len, 1, 1024]
        """
        self.lstm.flatten_parameters()

        # [seq_len, 1, hidden_size * 2]
        features, (h_n, c_n) = self.lstm(features)

        # [seq_len, 2]
        raw_logits = self.linear(features.squeeze(1))
        raw_scores = self.out(raw_logits/tau)
        
        # [seq_len, 1]
        scores = torch.matmul(raw_scores, label10).unsqueeze(1)
        
        #[seq_len, 1, hidden_size]
        weighted_features = original_features * scores.view(-1, 1, 1)
        
        #return raw probilities, gumbel hard/soft score, summary, gumbel soft score
        return scores, weighted_features