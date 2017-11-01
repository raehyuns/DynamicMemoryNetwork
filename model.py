import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import numpy as np

class DMNmodel(nn.Module):
    def __init__(self,config):
        super(DMNmodel,self).__init__()
        self.glove = config.glove_dim
        self.B = config.batch_size

        self.H_dim = config.hidden_dim
        self.z_dim = self.H_dim * 7 + 2
        
        self.m_update = config.memory_depth

        # num layer explicitly coded as 1
        self.GRU_i = nn.GRU(self.glove,self.H_dim,1,batch_first=True)
        
        self.GRU_q = nn.GRU(self.glove,self.H_dim,1,batch_first=True)

        self.att_W = Variable(torch.rand(self.H_dim,self.H_dim))
        self.att_gW1 = nn.Linear(self.z_dim,self.H_dim)
        self.att_gW2 = nn.Linear(self.H_dim,1)
        
        self.GRU_e = nn.GRU(self.H_dim,self.H_dim,1,batch_first=True)
        self.GRU_m = nn.GRU(self.H_dim,self.H_dim,1)

    def GetFactsRep(self,output_i,EOS_idx):
        # output_i (batch_size, seq_len, hidden_dim)
        # EOS_idx has EOS position of each input sequence
        # facts_rep (batch_size, max_s, hidden_dim)
        
        max_s = len(EOS_idx[0])
        batch_size = output_i.data.shape[0]
        hidden_dim = output_i.data.shape[2]

        facts_rep = torch.zeros([batch_size,max_s,hidden_dim])
        
        for ix in range(batch_size):
            num_s = len(EOS_idx[ix])
            
            # output_i shape = (nums,hidden_dim)
            facts_rep[ix,:num_s,:] = output_i[ix][EOS_idx[ix]].data
        
        return facts_rep
    
    def InputModule(self,input,seq_len,EOS_idx):
        # Input GRU module
        # Module's input should be packed first
        
        input = Variable(input,requires_grad=False) 
        packed_input = pack_padded_sequence(input,seq_len,batch_first=True)
        
        # output_i shape (B,seq_len,H_dim)
        packed_output, h = self.GRU_i(packed_input)
        output_i, _ = pad_packed_sequence(packed_output,batch_first=True)
         
        # facts_rep (B,max_s,H_dim)
        facts_rep = self.GetFactsRep(output_i,EOS_idx)

        return facts_rep

    def QuestionModule(self,Q,seq_len):
        # Question GRU module
        
        question = Variable(Q,requires_grad=False)
        packed_Q = pack_padded_sequence(question,seq_len,batch_first=True)

        # Here we only need last hidden state of GRU_q for Q_summary
        packed_output, Q_summary = self.GRU_q(packed_Q)
        output_q, _ = pad_packed_sequence(packed_output,batch_first=True)
        
        Q_summary = torch.squeeze(Q_summary)
        
        return Q_summary

    def Attention_Z(self,c,m,q):
        # c : fact representation at each time step (B,max_s,H_dim)
        # m : previous memory state (5,H_dim)
        # q : question summary (5,H_dim)
        
        col = self.H_dim * 7 + self.B * 2
        
        element = [c,m,q]

        element.append(torch.mul(c,q))
        element.append(torch.mul(c,m))
        element.append(torch.abs(c-q))
        element.append(torch.abs(c-m))
        # cWq & cWM output (B,1) 
        element.append(torch.diag((c.mm(self.att_W)).mm(q.transpose(0,1)))\
                                                            .unsqueeze(1))
        element.append(torch.diag((c.mm(self.att_W)).mm(m.transpose(0,1)))\
                                                            .unsqueeze(1))
        
        z = torch.cat(element,1) 
        
        return z

    def Attention_G(self,c,m,q):
        # output g is a similarity between c & q
        
        z = self.Attention_Z(c,m,q)
        
        H = F.tanh(self.att_gW1(z)) 
        g = F.sigmoid(self.att_gW2(H))
        
        return g
    
    def Attention(self,C,m,q):
        # get attention weight for every fact representation
        max_s = C.shape[1]
        g_list = []

        for i in range(max_s):
            c_t = Variable(C[:,i,:],requires_grad=False)
            g_t = self.Attention_G(c_t,m,q)
            
            g_list.append(g_t)

        attention = torch.cat(g_list,1)
        
        return attention

    def Modified_GRU(self,C,attention):
        
        max_l = C.shape[1]
        
        for i in range(max_l):
            # gate value for each time step (B,1)
            g_t = attention[:,i].unsqueeze(1)
            c_t = Variable(C[:,i,:].unsqueeze(1),requires_grad=False)

            if i ==0:
                ht_1 = Variable(torch.zeros(1,self.B,self.H_dim))
            else:
                ht_1 = h_t
            
            out,h_t = self.GRU_e(c_t,ht_1)
            
            #using attention weight g to get next time step h
            h_t = torch.mul(h_t.squeeze(),g_t) + torch.mul((1-g_t),ht_1)
        
        return h_t

    def MemoryUpdate(self,C,Q):
        for i in range(self.m_update):
            
            if i==0:
                Mt_1 = Q
            else:
                Mt_1 = Mt.squeeze()

            attention = self.Attention(C,Q,Mt_1)
            h_t = self.Modified_GRU(C,attention)
            
            out,Mt = self.GRU_m(h_t,Mt_1.unsqueeze(0))
            
        return Mt 

   
















