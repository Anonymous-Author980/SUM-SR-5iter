import torch
import torch.nn as nn

def random_drop(features, p, device):
    '''
    randomly dropped 1-p features from the original features
    features: [seq_len, d_model]
    p: between 0-1
    device: device that features on
    '''

    seq_len= features.size()[0]
    probs = torch.ones(seq_len)
    weights = torch.zeros(seq_len)
    idxs = torch.multinomial(probs, seq_len)
    keep_idx = idxs[:int(seq_len*p)]
    mask_idx = idxs[int(seq_len*p):]
    weights[keep_idx] = 1.0
    weights = weights.to(device)
    output = features * weights.unsqueeze(1)

    return output, weights.type(torch.LongTensor), mask_idx




