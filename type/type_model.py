import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm
import pickle


class Transfer_Type(nn.Module):
    def __init__(self, args, base_model):
        super(Transfer_Type, self).__init__()
        self.args = args
        self.base_model = base_model
        self.final_layer = torch.nn.Sequential(
            nn.Linear(self.args.hidden_size * 2, 300),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(300, 2)    
        )


    def forward(self, alias, alias_word_mask, alias_char_mask):
        alias_rep = self.base_model.char_encoder(alias, alias_char_mask)
        alias_rep = self.base_model.word_encoder(alias_rep, alias_word_mask)
        score = self.final_layer(alias_rep)
        return score


class Alias_Type(nn.Module):
    '''
    Model for Type detection
    '''
    def __init__(self, args, voc, ind2char, char2ind):
        super(Alias_Type, self).__init__()
        self.args = args
        self.voc = voc
        self.ind2char = ind2char
        self.char2ind = char2ind

        self.embeddings = nn.Embedding(num_embeddings=len(voc), embedding_dim=self.args.embedding_dim,
                                        padding_idx=0, max_norm=None, scale_grad_by_freq=False, sparse=False)

        self.char_rnn = Encoder_rnn(args, self.args.input_size, self.args.hidden_size)
        self.char_self_attn = LinearAttn(self.args.hidden_size * 2)
        self.linear_layer = torch.nn.Sequential(
            nn.Linear(self.args.hidden_size * 2, 600),
            nn.Dropout(0.5),
            nn.ReLU(),
        )
        self.word_rnn = Encoder_rnn(args, self.args.input_size * 2, self.args.hidden_size)
        self.word_self_attn = LinearAttn(self.args.hidden_size * 2)
        for i in range(self.args.num_classes):
            setattr(self, 'final_layer_' + str(i), torch.nn.Sequential(
            nn.Linear(self.args.hidden_size * 2, 300),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(300, 2)    
        ))
        
    
    def char_encoder(self, alias, alias_char_mask):
        #device = torch.device("cuda" if self.args.cuda else "cpu")
        device = torch.device("cpu")
        
        batch_size, word_max_len, char_max_len = alias.size(0), alias.size(1), alias.size(2)
        alias = alias.view(batch_size * word_max_len, char_max_len)   
        alias_rep = self.embeddings(alias)
        

        alias_char_len = alias_char_mask.view(batch_size * word_max_len, char_max_len).data.eq(0).long().sum(1).cpu().numpy().tolist()
        alias_char_mask = alias_char_mask.view(batch_size * word_max_len, char_max_len)

        alias_char_idx = [i for i in range(len(alias_char_len)) if alias_char_len[i] > 0]
        alias_char_mask = alias_char_mask.view(batch_size * word_max_len, char_max_len)[alias_char_idx]
        alias_char_len = [i for i in alias_char_len if i > 0]
        alias_rep = alias_rep[alias_char_idx]

        alias_rep = self.char_rnn(alias_rep, alias_char_len)  
        alias_rep = self.char_self_attn(alias_rep, alias_char_mask)  
        alias_word_rep = torch.FloatTensor(batch_size * word_max_len, 2 * self.args.hidden_size).zero_().to(device)
        alias_word_rep[alias_char_idx] = alias_rep
        
        return alias_word_rep.view(batch_size, word_max_len, -1)
    
    def word_encoder(self, alias, alias_word_mask, is_sample=False):
        alias_word_len = alias_word_mask.data.cpu().eq(0).long().sum(1).cpu().numpy().tolist()
        alias_rep = self.word_rnn(alias, alias_word_len, is_sample)
        alias_rep = self.word_self_attn(alias_rep, alias_word_mask)
        return alias_rep
    

    def forward(self, alias, alias_word_mask, alias_char_mask):
        '''
        Forward pass of type detection model
        char_encoder -> word_encoder -> MLP
        '''
        alias_rep = self.char_encoder(alias, alias_char_mask)
        alias_rep = self.word_encoder(alias_rep, alias_word_mask)
        score_list = list()
        for i in range(self.args.num_classes):
            res = getattr(self, 'final_layer_' + str(i))(alias_rep)
            score_list.append(res)
      
        return score_list



class Encoder_rnn(nn.Module):
    '''
    Encoder layer (GRU)
    '''
    def __init__(self, args, input_size, hidden_size):
        super(Encoder_rnn, self).__init__()
        self.args = args

        self.rnn = nn.GRU(input_size = input_size,
                          hidden_size = hidden_size,
                          num_layers = self.args.num_layers,
                          batch_first = True,
                          dropout = self.args.dropout,
                          bidirectional = self.args.bidirect)

    

    def forward(self, alias, alias_len, is_sample=False):  

        alias_len = np.array(alias_len)
        sorted_idx= np.argsort(-alias_len)
        alias = alias[sorted_idx]
        alias_len = alias_len[sorted_idx]
        unsorted_idx = np.argsort(sorted_idx)

        packed_emb = torch.nn.utils.rnn.pack_padded_sequence(alias, alias_len, batch_first=True)
        output, hn = self.rnn(packed_emb)
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(output)
        unpacked = unpacked.transpose(0, 1)
        unpacked = unpacked[torch.LongTensor(unsorted_idx)]
        return unpacked
       



class LinearAttn(nn.Module):
    def __init__(self, input_size):
        super(LinearAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        self-attention Layer

        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
        x_flat = x.view(x.size(0) * x.size(1), x.size(2))
        ##### change to batch * len * hdim
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
      
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=1)

        #x = x.transpose(0, 1)
        output_avg = alpha.unsqueeze(1).bmm(x).squeeze(1)

        return output_avg