import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle
import numpy as np
from dataset import *
from torch.nn.utils.rnn import pack_padded_sequence


def run_train(data,embed,m,config):
    w_idx = LoadPickle('w_idx.txt')
    
    for epoch in range(config.epoch):
        epoch_Aloss = epoch_Gloss = []
        epoch_Acorrect = epoch_Gcorrect  = 0
        num_Q = num_G = 0
        
        for i in range(len(data)//config.batch_size):
            data_pair = data[i*config.batch_size:(i+1)*config.batch_size]
            max_i, max_q, max_s = FindLongest(data_pair)
            
            # split data_pair into i,q,a
            input_data = ([p[0] for p in data_pair])
            question_data = ([p[1] for p in data_pair])
            answer_data = ([p[2] for p in data_pair])
            attention_data = ([p[3] for p in data_pair])
             
            # Descending order input, seq_length, EOS position 
            input,seq_len_i,EOS_idx = GetEmbedding(embed,input_data,max_i)  
            question,seq_len_q,_ =  GetEmbedding(embed,question_data,max_q)
            
            Att, m_depth = GetAttSent(attention_data)
            
            A = GetAnsweridx(answer_data,embed)
            C = m.InputModule2(input,seq_len_i,EOS_idx)
            Q = m.QuestionModule2(question,seq_len_q)
            
            Mtm, att_W = m.MemoryUpdate(C,Q,EOS_idx,m_depth,config.m_mode)
            pred = m.AnswerModule(Mtm,Q,A)
            
            '''
            if i%100 == 0 :
                print(att_W[0])
                print(att_W[-1])
                print(i)
            ''' 
            m.optimizer.zero_grad()
            
            # Get both Answer & Gate loss to report
            Aloss, Acorrect = m.GetAnswerLoss(pred,A)
            Gloss, Gcorrect = m.GetGateLoss(att_W,Att,m_depth)
            
            # Choose bewtween which loss to minimize
            if config.train_mode == 'A':
                loss = Aloss + Gloss
            else:
                loss = Gloss

            epoch_Aloss.append(Aloss.data)
            epoch_Gloss.append(Gloss.data)
            epoch_Acorrect += Acorrect
            epoch_Gcorrect += Gcorrect
            num_Q += pred.data.shape[0]
            num_G += np.sum([len(D) for D in attention_data])

            if i%2 ==0:
                #print(att_W[0])
                num_G = m_depth * num_Q
                _progress = "\r(Epoch %d)"%epoch
                _progress += "training step %i"%i
                _progress += "||| Answer Loss : %.4f"%(Aloss.data[0])
                _progress += " | accuracy : %.4f"%(epoch_Acorrect/(num_Q))
                _progress += " | total correct %d out of %d"%(epoch_Acorrect,num_Q)
                _progress += " ||| Gate Loss : %.4f"%(Gloss.data[0])
                _progress += " | accuracy : %.4f"%(epoch_Gcorrect/num_G)
                _progress += " | total correct %d out of %d"%(epoch_Gcorrect,num_G)

                sys.stdout.write(_progress)
                sys.stdout.flush()
            
            loss.backward()
            m.optimizer.step()
            
        print('\r\n\n##### Epoch %d Average Answer Loss : %.4f &  Accuracy %.4f'%\
        (epoch,np.mean(epoch_Aloss)[0],epoch_Acorrect/((i+1)*config.batch_size)))
        
        print('\n Average Gate Loss : %.4f & Accuracy %.4f####'%(
        np.mean(epoch_Gloss)[0],epoch_Gcorrect/num_G))

    torch.save(m.state_dict(),
                    '/home/raehyun/github/DMN/model/%s'%config.save_model)
        
def run_test(data,embed,m,config):
    w_idx = LoadPickle('w_idx.txt')
    
    epoch_Aloss = epoch_Gloss = []
    epoch_Acorrect = epoch_Gcorrect  = 0
    num_Q = 0
    
    for i in range(len(data)//config.batch_size):
        data_pair = data[i*config.batch_size:(i+1)*config.batch_size]
        max_i, max_q, max_s = FindLongest(data_pair)
        
        # split data_pair into i,q,a
        input_data = list(reversed([p[0] for p in data_pair]))
        question_data = list(reversed([p[1] for p in data_pair]))
        answer_data = list(reversed([p[2] for p in data_pair]))
        attention_data = list(reversed([p[3] for p in data_pair]))
        
        # Descending order input, seq_length, EOS position 
        input,seq_len_i,EOS_idx = GetEmbedding(embed,input_data,max_i)  
        question,seq_len_q,_ =  GetEmbedding(embed,question_data,max_q)
        
        Att, memory_depth = GetAttSent(attention_data)
        
        A = GetAnsweridx(answer_data,embed)
        C = m.InputModule(input,seq_len_i,EOS_idx)
        Q = m.QuestionModule(question,seq_len_q)
         
        Mtm, att_W = m.MemoryUpdate(C,Q,EOS_idx,memory_depth)
        pred = m.AnswerModule(Mtm,Q,A)
        
        m.optimizer.zero_grad()
        
        # Get both Answer & Gate loss to report
        Aloss, Acorrect = m.GetAnswerLoss(pred,A)
        Gloss, Gcorrect = m.GetGateLoss(att_W,Att,memory_depth)
        
        # Choose bewtween which loss to minimize
        if config.train_mode == 'A':
            loss = Aloss + Gloss
        else:
            loss = Gloss

        epoch_Aloss.append(Aloss.data)
        epoch_Gloss.append(Gloss.data)
        epoch_Acorrect += Acorrect
        epoch_Gcorrect += Gcorrect
        num_Q += pred.data.shape[0]
        
        if i%2==0:
            _progress = "\rtesting step %i "%i
            _progress += "||| Answer Loss : %.4f"%(Aloss.data[0])
            _progress += " | accuracy : %.4f"%(epoch_Acorrect/(num_Q))
            _progress += " | total correct %d out of %d"%(epoch_Acorrect,num_Q)
            _progress += " ||| Gate Loss : %.4f"%(Gloss.data[0])
            _progress += " | accuracy : %.4f"%(epoch_Gcorrect/(num_Q))
            _progress += " | total correct %d out of %d"%(epoch_Gcorrect,num_Q)

            sys.stdout.write(_progress)
            sys.stdout.flush()
        
    result = '\r\n\n##### Average Answer Loss : %.4f &  Accuracy %.4f'%(
        np.mean(epoch_Aloss)[0],epoch_Acorrect/((i+1)*config.batch_size))
    result +=  '  || Average Gate Loss : %.4f & Accuracy %.4f #####\n'%(
        np.mean(epoch_Gloss)[0],epoch_Gcorrect/((i+1)*config.batch_size))

    print(result)

