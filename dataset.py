import os
import pickle
import nltk
from nltk.tokenize import RegexpTokenizer
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

def IndexChange(data):
    index_dict = {}
    temp = []
    value = N_q = 0

    for key,line in enumerate(data):
        if '?' in line:
            continue
        else:
            index_dict[key+1] = value
            value +=1
    
    return index_dict

def _createQApair(data):
    # Given one group of lines, create list of [i,q,a,att]
    index_dict = IndexChange(data)
    
    qTokenizer = RegexpTokenizer(r'\w+')
    pairs = []
    i = []
    
    for line in data:
        if '?' in line:
            split_line = line.split(sep='\t')
            tokens = [qTokenizer.tokenize(l) for l in split_line]
            
            q = [t.lower() for t in tokens[0][1:]]
            a = [t.lower() for t in tokens[1]]
            att = [index_dict[int(t)] for t in tokens[2]]
            
            pairs.append([i[:],q,a,att])

        else:
            tokens = nltk.word_tokenize(line)
            tokens = [token.lower() for token in tokens]
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
    # Pairs[0](QAset) -- list of [i,q,a,att] 
    print('Loading Data')
    file = os.path.join(data_dir,filename)
    f = open(file,'r')
    lines = f.readlines()

    Dataset = SplitDataset(lines)
    Pairs = CreateQAPair(Dataset)
    print('Total Sentence %d'%len(Pairs))
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

def FindMaxS(EOS_idx):
    maxs = 0
    for idx in EOS_idx:
        if len(idx) > maxs:
            maxs = len(idx)
    return maxs

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

def GetEmbedding(glove,data,max_l,dtype='input'):
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
    ''' 
    if dtype == 'input':
        # sorting vector by its length (descending order)
        seq_length = torch.LongTensor(seq_length)
        seq_length, sort_idx = seq_length.sort(0,descending=True)
        seq_tensor = seq_tensor[sort_idx]
        
        EOS = FindEOS(data) 
        EOS_idx = [EOS[i] for i in sort_idx]
        return seq_tensor,seq_length.numpy(),EOS_idx 
    
    elif dtype == 'question':
    '''
    return seq_tensor,np.array(seq_length),EOS_idx


def FindEOS(input_data):
    # find EOS index for each input data
    EOS_list = []
    for input_seq in input_data: 
        idxs = torch.LongTensor([ix for ix,w in enumerate(input_seq) if w=='.'])
        
        EOS_list.append(idxs)
    
    return EOS_list

def MakeDBDict(data_dir):
    files = os.listdir(data_dir)
    total_word = set()

    for filename in files:
        if 'txt' in filename:
            file = os.path.join(data_dir,filename)
            f = open(file,'r')
            lines = f.readlines()
            for line in lines:
                tokens = nltk.word_tokenize(line)
                token = [w.lower() for w in tokens if not w.isdigit()]        
                for w in token:
                    total_word.add(w)
    print(len(total_word))
    
    with open('total_word.txt','wb') as f:
        pickle.dump(total_word,f)
    print('Word Dicts are saved')
    return total_word

def WordEmbed(word_dict):
    glove = GloveVector(300)
    wordEmbed = glove.loc[word_dict]
    wordEmbed.to_csv('wordembed.csv')
    
    return wordEmbed

def MaxLength(data):
    M = 0
    for d in data:
        if len(d) > M:
            M = len(d)
    return M

def GetAnsweridx(answer_data,word_embed):
    # Get answer words' index values
    # target_idx [B,Num of Answers] 
    
    max_a = MaxLength(answer_data)
    
    target_idx = []
    
    for i,data in enumerate(answer_data):
        indexes = []
        for j,answer_word in enumerate(data):
            idx = word_embed.index.get_loc(answer_word)
            indexes.append(idx)         
        indexes = torch.LongTensor(indexes).unsqueeze(0)
        target_idx.append(indexes)
    target_idx = torch.cat(target_idx,0)
    return target_idx


def LoadPickle(filename):
    with open(filename,'rb') as f:
        load = pickle.load(f)
    return load

def GetAttSent(attention_data):
    # Get index value of sentence model should focus to answer
    # attention_data is a list of indexes lists
    max_depth = 0
    out = [] 
    for att in attention_data:
        if len(att) > max_depth:
            max_depth = len(att)
    
    for n,att in enumerate(attention_data):
        diff = max_depth - len(att)
        int_att = [int(ix) for ix in att]
        if diff > 0:
            int_att += [int_att[-1]]*diff
        indexes = torch.LongTensor(int_att).unsqueeze(0)
        
        out.append(indexes)
    
    out = torch.cat(out,0) 
    
    return out, max_depth









































