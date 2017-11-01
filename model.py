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
        self.B = config.batch_size
        self.H_dim = config.hidden_dim

        self.GRU_i = nn.GRU(config.glove_dim,config.hidden_dim,
                                config.num_layer,batch_first=True)
        
        self.GRU_q = nn.GRU(config.glove_dim,config.hidden_dim,
                                config.num_layer,batch_first=True)

        self.att_W = Variable(torch.rand(config.hidden_dim,config.hidden_dim))
        self.att_linear = nn.Linear()                                                    
    
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
        # cWq & cWM output (B,5) 
        element.append((c.mm(self.att_W)).mm(q.transpose(0,1)))
        element.append((c.mm(self.att_W)).mm(m.transpose(0,1)))
        
        z = torch.cat(element,1) 
        
        return z

    def Attention_G(self,c,m,q):
        
        z = self.Attention_Z(c,m,q)
        ##### G function dimension check


























