import os
import nltk
import numpy as np
import pandas as pd
import csv
import torch

def SplitDataset(lines):
    # Split dataset by number
    # Make new group when line # 1 found 
    
    Dataset = []
    check_s = check_e = 0
    
    for i,line in enumerate(lines):
        tokens = nltk.word_tokenize(line)
        if '?' in tokens:
            q_t = tokens.index('?')
            if i+1 == len(lines):
                Dataset.append(lines[check_s:])
            elif nltk.word_tokenize(lines[i+1])[0]=='1':
                check_e = i+1
                Dataset.append(lines[check_s:check_e])
                check_s = check_e
    
    return Dataset

def _createQApair(data):
    # Given one group of lines, create list of [i,q,a]

    pairs = []
    i = []

    for line in data:
        tokens = nltk.word_tokenize(line)
        tokens = [token.lower() for token in tokens]
        if '?' in tokens:
            q_t = tokens.index('?') + 1
            q = tokens[1:q_t]
            a = tokens[q_t:]
            pairs.append([i[:],q,a])
        else:
            i += tokens[1:]
    
    return pairs

def CreateQAPair(dataset):
    # Given total dataset, create QA pairs for each data using _createQApair
    # len(pairs) = (Num of paragraph * Num of question)
    pairs = []

    for d in dataset:
        pair = _createQApair(d)
        pairs += (pair)
    
    return pairs

def GetRawData(data_dir,filename):
    # Pairs -- list of QAset
    # Pairs[0](QAset) -- list of [i,q,a] 

    file = os.path.join(data_dir,filename)
    f = open(file,'r')
    lines = f.readlines()

    Dataset = SplitDataset(lines)
    Pairs = CreateQAPair(Dataset)

    return Pairs

def FindLongest(pairs):
    max_i = 0
    max_q = 0
    max_s = 0

    for pair in pairs:
        if len(pair[0]) > max_i:
            max_i = len(pair[0])
        if len(pair[1]) > max_q:
            max_q = len(pair[1])
        if pair[0].count('.') > max_s:
            max_s = pair[0].count('.')

    return max_i,max_q,max_s


def GloveVector(dim):
    # Create Matrix with word vectors (N,dim)
    print("Loading Glove Vector")
    filename = 'glove.6B.%dd.txt'%dim
    file = os.path.join(r'./glove',filename)
    
    words_matrix = pd.read_table(file,sep=" ",index_col=0,
                        header=None,quoting=csv.QUOTE_NONE)
    print("Glove Vector loading complete!!")
    return words_matrix

def vec(glove,w):
    return glove.loc[w].as_matrix()

def GetEmbedding(glove,data,max_l):
    # Initialize zero tensor size (B,max_l,D)
    batch_size = len(data)
    glove_dim = glove.shape[1]
    seq_tensor = torch.zeros([batch_size,max_l,glove_dim])
    seq_length = []
    
    for i,words in enumerate(data):
        # Get embedding for each word in sequence
        word_vectors = [vec(glove,w) for w in words]
        word_vectors = torch.FloatTensor(np.stack(word_vectors))
        
        seq_len = word_vectors.shape[0]
        
        seq_length.append(seq_len)
        seq_tensor[i,:seq_len,:] = word_vectors
    
    EOS_idx = FindEOS(data) 
    
    # sorting vector by its length (descending order)
    seq_length = torch.LongTensor(seq_length)
    seq_length, sort_idx = seq_length.sort(0,descending=True)
    seq_tensor = seq_tensor[sort_idx]
    
    EOS = FindEOS(data) 
    EOS_idx = [EOS[i] for i in sort_idx]
    return seq_tensor,seq_length.numpy(),EOS_idx

def FindEOS(input_data):
    # find EOS index for each input data
    EOS_list = []
    for input_seq in input_data: 
        idxs = torch.LongTensor([ix for ix,w in enumerate(input_seq) if w=='.'])
        
        EOS_list.append(idxs)
    
    return EOS_list























