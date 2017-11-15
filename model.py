import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import numpy as np
from dataset import *

class DMNmodel(nn.Module):
    def __init__(self,config):
        super(DMNmodel,self).__init__()
        
        self.glove = config.glove_dim
        self.B = config.batch_size

        self.H_dim = config.shareH_dim
        self.gW_dim = config.gW_dim
        self.z_dim = self.H_dim * 7 + 2 
        
        self.GRU_i = nn.GRU(self.glove,self.H_dim,config.num_layer,
                                                    batch_first=True)
        
        self.GRU_q = nn.GRU(self.glove,self.H_dim,config.num_layer,
                                                    batch_first=True)

        self.att_zW = nn.Linear(self.H_dim,self.H_dim,bias=False) 

        self.att_gW1 = nn.Linear(self.z_dim,self.gW_dim)
        self.att_gW2 = nn.Linear(self.gW_dim,1)
        
        self.GRUcell_e = nn.GRUCell(self.H_dim,self.H_dim)
        self.GRUcell_m = nn.GRUCell(self.H_dim,self.H_dim)
       
        self.Wa_linear = nn.Linear(self.H_dim,159,bias=False)
        self.GRUcell_a = nn.GRUCell(self.H_dim+159,self.H_dim)

        self.loss_criterion = nn.CrossEntropyLoss()
        
        self.optimizer = optim.Adam(self.parameters(),lr=config.lr)

    def GetFactsRep(self,output_i,EOS_idx):
        # output_i (batch_size, seq_len, hidden_dim)
        # EOS_idx has EOS position of each input sequence
        # facts_rep (batch_size, max_s, hidden_dim)
         
        max_s = FindMaxS(EOS_idx) 
        batch_size = output_i.data.shape[0]
        hidden_dim = output_i.data.shape[2]
        
        facts = []

        for ix in range(batch_size):
            num_s = len(EOS_idx[ix])
            diff = max_s - num_s
            if diff > 0 :
                pad = Variable(torch.zeros(diff,hidden_dim))
            # output_i shape = (nums,hidden_dim)

            valid = output_i[ix][EOS_idx[ix].cuda()]
            
            if diff > 0 :
                pad = Variable(torch.zeros(diff,hidden_dim)).cuda()
                valid = torch.cat([valid,pad])
            
            facts.append(valid.unsqueeze(0))
        
        facts_rep = torch.cat(facts)
        
        return facts_rep
    
    def InputModule(self,input,seq_len,EOS_idx):
        # Input GRU module
        # Module's input should be packed first
        
        input = Variable(input,requires_grad=False).cuda() 
        
        packed_input = pack_padded_sequence(input,seq_len,batch_first=True)
        
        # output_i shape (B,seq_len,H_dim)
        packed_output, h = self.GRU_i(packed_input)
        output_i, _ = pad_packed_sequence(packed_output,batch_first=True)
         
        # facts_rep (B,max_s,H_dim)
        facts_rep = self.GetFactsRep(output_i,EOS_idx)
        
        return facts_rep
    
    def InputModule2(self,input,seq_len,EOS_idx): 
        
        input = Variable(input,requires_grad=False).cuda()
        
        output,_ = self.GRU_i(input)
        facts_rep = self.GetFactsRep(output,EOS_idx)
        
        return facts_rep

    def QuestionModule2(self,Q,seq_len):
        # Q [B,max_w,H]
        question = Variable(Q,requires_grad=False).cuda()
        output, _ = self.GRU_q(question)
        
        Q_summary = []
        
        for i in range(Q.shape[0]):
            q_len = seq_len[i] - 1
            Q_summary.append(output[i,q_len,:].unsqueeze(0))
        
        Q_summary = torch.cat(Q_summary,0)
        
        return Q_summary
    
    def QuestionModule(self,Q,seq_len):
        # Question GRU module
        
        question = Variable(Q,requires_grad=False).cuda()
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
        
        element = [c,m,q]
        #element = []
        
        element.append(torch.mul(c,q))
        element.append(torch.mul(c,m))
        element.append(torch.abs(c-q))
        element.append(torch.abs(c-m))
        # cWq & cWM output (B,1) 
        
        cW = self.att_zW(c)
        element.append(torch.diag(cW.mm(q.transpose(0,1))).unsqueeze(1))
        element.append(torch.diag(cW.mm(m.transpose(0,1))).unsqueeze(1))
        
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
        max_s = C.data.shape[1]
        g_list = []

        for i in range(max_s):
            c_t = C[:,i,:]
            g_t = self.Attention_G(c_t,m,q)
            
            g_list.append(g_t)

        attention = torch.cat(g_list,1)
        
        return attention

    def Modified_GRU(self,C,attention,seq_len):
        max_l = C.data.shape[1]
        ht_list = [] 
        
        for i in range(max_l):
            # gate value for each time step (B,1)
            g_t = attention[:,i].unsqueeze(1)
            c_t = C[:,i,:]
            
            if i ==0:
                ht_1 = Variable(torch.zeros(self.B,self.H_dim)).cuda()
            else:
                ht_1 = h_t
            
            h_t = self.GRUcell_e(c_t,ht_1)
            
            #using attention weight g to get next time step h
            h_t = torch.mul(h_t,g_t) + torch.mul((1-g_t),ht_1)
            ht_list.append(h_t.unsqueeze(1))
        
        Htc = self.GetLastState(ht_list,seq_len)
        
        return Htc
    
    def GetLastState(self,ht_list,EOS_idx):
        # ht_seq = [B,max_s,H_dim]
        # Htc = [B,H_dim] <-- Picked last hidden state
        ht_seq = torch.cat(ht_list,1)
        
        Htc = []
        for i in range(ht_seq.data.shape[0]):
            Htc.append(ht_seq[i][len(EOS_idx[i])-1].unsqueeze(0))
        
        Htc = torch.cat(Htc,0)
        
        return Htc

    def SoftmaxRep(self,C,attention):
        # attention [B,max_S] / C [B,max_s,H]
        soft_att = F.softmax(attention)
        
        B = soft_att.data.shape[0]
        C = C.transpose(1,2)
        H = Variable(torch.zeros(C.data.shape)).cuda()
        
        for i in range(B):
            att_Bi = soft_att[i]
            H[i] = C[i] * att_Bi
        
        Htc = H.sum(2)
        
        return Htc

    def MemoryUpdate(self,C,Q,EOS_idx,m_depth,mode='softmax'):
        # C [B,Max_s,H_dim]
        # Q [B,H_dim]
        Attentions = [] 
        for i in range(m_depth):
            
            if i==0:
                Mt_1 = Q
            else:
                Mt_1 = Mt
            
            # Htc : Last Hidden State after Attention weighted GRU
            attention = self.Attention(C,Q,Mt_1)
            
            Attentions.append(attention.unsqueeze(1))
            
            # Simple Softmax function or Modified GRU to get final episode(Htc)
            if mode == 'softmax':
                Htc = self.SoftmaxRep(C,attention) 
            else:
                Htc = self.Modified_GRU(C,attention,EOS_idx)
            
            #GRU_m is vertical GRU in visualization of model to update memory
            Mt = self.GRUcell_m(Htc,Mt_1)
        
        Attentions = torch.cat(Attentions,1)
        
        return Mt,Attentions 
    
    def AnswerModule(self,Mtm,q,target):
        # With last Memory state(Mtm) Create Answer seq.
        # pred [B,N_answer,159]
        preds = [] 
        
        at_1 = Mtm
        yt_1 = F.softmax(self.Wa_linear(at_1))
        
        a_len = target.shape[1]

        for i in range(a_len):
            xt = torch.cat([yt_1,q],1)
            at = self.GRUcell_a(xt,at_1)
            
            yt = F.softmax(self.Wa_linear(at))
            
            preds.append(yt)
            
            at_1 = at
            yt_1 = yt

        pred = torch.cat(preds,1)

        return pred

    def GetAnswerLoss(self,pred,target):
        # target [B, N of answers]
        # pred [B, 159] 
        N_answer = target.shape[1]
        target = Variable(target,requires_grad=False).cuda()
        
        if N_answer == 1 :
            target = target.squeeze()
            Aloss = self.loss_criterion(pred,target)
            correct  = self.GetCorrect(pred,target)
            
            return Aloss,correct

        else:
            # targets shape [N of As,B]
            # preds shape [N of As,5,159]
            targets = target.transpose(0,1)
            preds = pred.view(-1,159,N_answer)
            total_loss = 0
            total_accuracy = 0

            for i in range(N_answer):
                pred = preds[:,:,i]
                target = targets[i].squeeze()
                loss = self.loss_criterion(pred,target)
                correct = self.GetCorrect(pred,target)
                
                total_loss += loss
                total_correct += correct
            
            return total_loss,total_correct 
    
    def GetGateLoss(self,att_W,Att,memory_depth):
        # att_W : [B, m_depth, max_s] / Att : [B,m_depth]
        # We only need focus single sentence to answer
        if memory_depth==1:
            att_W = att_W.squeeze() 
            target = Variable(Att.squeeze(),requires_grad=False).cuda()
            
            Gloss = self.loss_criterion(att_W,target)
            correct = self.GetCorrect(att_W,target)
            
            return Gloss, correct
        
        # Need Several Memory Update
        else:
            Gloss = correct = 0
            for depth in range(memory_depth):
                #att_w [B,max_s] / target [B]
                att_w = att_W[:,depth,:].squeeze()
                target = Variable(Att[:,depth].squeeze(),
                                            requires_grad=False).cuda()
                
                Gloss += self.loss_criterion(att_w,target)
                
                correct += self.GetCorrect(att_w,target)
        
            return Gloss,correct

    def GetCorrect(self,pred,target):
        _,pred_ix = torch.max(pred,1)
        
        correct = (pred_ix==target).sum()

        return correct.data[0]

        




