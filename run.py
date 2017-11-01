import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from dataset import *
from torch.nn.utils.rnn import pack_padded_sequence


def run_train(data,glove,m,config):
    for i in range(len(data)//config.batch_size):
        data_pair = data[i*config.batch_size:(i+1)*config.batch_size]
        max_i, max_q, max_s = FindLongest(data_pair)
        
        # split data_pair into i,q,a
        input_data = [p[0] for p in data_pair]
        question_data = [p[1] for p in data_pair]
        answer_data = [p[2] for p in data_pair]
        
        # Descending order input, seq_length, EOS position 
        input,seq_len_i,EOS_idx = GetEmbedding(glove,input_data,max_i)  
        question,seq_len_q,_ =  GetEmbedding(glove,question_data,max_q)
        
        
        C = m.InputModule(input,seq_len_i,EOS_idx)
        Q = m.QuestionModule(question,seq_len_q)
        m.Attention_Z(Variable(C[:,0,:],requires_grad=False),Q,Q)
        
        break
