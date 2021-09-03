import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
from torch.nn import functional as F


class Hybrid_Alias_Sim(nn.Module):
    def __init__(self, args, voc):
        super(Hybrid_Alias_Sim, self).__init__()
        self.args = args
        self.char_rnn = Encoder_rnn(args, self.args.input_size, self.args.hidden_size)
        self.char_match_attn = MatchAttn(self.args.hidden_size * 2)
        self.char_linear_attn = LinearAttn(self.args.hidden_size * 4)
        self.char_linear = torch.nn.Sequential(
            nn.Linear(self.args.hidden_size * 4, self.args.hidden_size * 2),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        #out_hidden_size = 2 * self.args.hidden_size if self.args.bidirect else self.args.hidden_size
        self.word_rnn = Encoder_rnn(args, self.args.input_size * 2, self.args.hidden_size)
        self.word_match_attn = MatchAttn(self.args.hidden_size * 2)
        self.word_self_attn = LinearAttn(self.args.hidden_size * 4)
        self.embeddings = nn.Embedding(num_embeddings=len(voc), embedding_dim=self.args.embedding_dim,
                                        padding_idx=0, max_norm=None, scale_grad_by_freq=False, sparse=False)

        self.final_layer = torch.nn.Sequential(
            nn.Linear(self.args.hidden_size * 8, 300),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(300, 1)    
        )
    
    def char_encoder(self, alias, alias_char_mask, is_sample=False):
        
        device = torch.device("cuda")
        batch_size, word_max_len, char_max_len = alias.size(0), alias.size(1), alias.size(2)
        alias = alias.view(batch_size * word_max_len, char_max_len)   
        alias_rep = self.embeddings(alias)

        alias_char_len = alias_char_mask.view(batch_size * word_max_len, char_max_len).data.eq(0).long().sum(1).cpu().numpy().tolist()
        alias_char_mask = alias_char_mask.view(batch_size * word_max_len, char_max_len)

        alias_char_idx = [i for i in range(len(alias_char_len)) if alias_char_len[i] > 0]
        alias_char_mask = alias_char_mask.view(batch_size * word_max_len, char_max_len)[alias_char_idx]
        alias_char_len = [i for i in alias_char_len if i > 0]
        alias_rep = alias_rep[alias_char_idx]

        alias_rep = self.char_rnn(alias_rep, alias_char_len, is_sample)
        if is_sample:
            alias_word_rep = torch.FloatTensor(batch_size * word_max_len, 2 * self.args.hidden_size).zero_().to(device)
            alias_word_rep[alias_char_idx] = alias_rep
            return alias_word_rep.view(batch_size, word_max_len, 2 * self.args.hidden_size)
        
        alias_word_rep = torch.FloatTensor(batch_size * word_max_len, char_max_len, 2 * self.args.hidden_size).zero_().to(device)
        
        
        alias_word_rep[alias_char_idx] = alias_rep
        
        return alias_word_rep.view(batch_size, word_max_len, char_max_len, -1)
    

    def char_self_attn(self, alias_rep, alias_char_mask):
        alias_char_len = alias_char_mask.data.eq(0).long().sum(1).cpu().numpy().tolist()
        alias_char_idx = [i for i in range(len(alias_char_len)) if alias_char_len[i] > 0]
        alias_char_mask = alias_char_mask[alias_char_idx]
        alias_rep = alias_rep[alias_char_idx]

        alias_rep = self.char_linear_attn(alias_rep, alias_char_mask)
        

        return alias_rep, alias_char_idx


    def char_attn(self, a1_rep, alias1_char_mask, a2_rep, alias2_char_mask):
        device = torch.device("cuda")
        batch_size, word_max_len1, char_max_len1 = a1_rep.size(0), a1_rep.size(1), a1_rep.size(2)
        a1_rep = a1_rep.view(batch_size, word_max_len1 * char_max_len1, -1)
        alias1_char_mask = alias1_char_mask.view(batch_size, word_max_len1 * char_max_len1)

        word_max_len2, char_max_len2 = a2_rep.size(1), a2_rep.size(2)
        a2_rep = a2_rep.view(batch_size, word_max_len2 * char_max_len2, -1)
        alias2_char_mask = alias2_char_mask.view(batch_size, word_max_len2 * char_max_len2)

        a1_attn_a2 = self.char_match_attn(a1_rep, a2_rep, alias2_char_mask)
        a2_attn_a1 = self.char_match_attn(a2_rep, a1_rep, alias1_char_mask)


        a1_rep = torch.cat((torch.abs(a1_rep - a1_attn_a2), torch.mul(a1_rep, a1_attn_a2)), 2)
        a2_rep = torch.cat((torch.abs(a2_rep - a2_attn_a1), torch.mul(a2_rep, a2_attn_a1)), 2)
        

        a1_rep, a1_idx = self.char_self_attn(a1_rep.view(batch_size * word_max_len1, char_max_len1, -1), alias1_char_mask.view(batch_size * word_max_len1, char_max_len1))
        a2_rep, a2_idx = self.char_self_attn(a2_rep.view(batch_size * word_max_len2, char_max_len2, -1), alias2_char_mask.view(batch_size * word_max_len2, char_max_len2))

        a1_rep = self.char_linear(a1_rep)
        a2_rep = self.char_linear(a2_rep)



        a1_word_rep = torch.FloatTensor(batch_size * word_max_len1, self.args.hidden_size * 2).zero_().to(device)
        a1_word_rep[a1_idx] = a1_rep
        a1_word_rep = a1_word_rep.view(batch_size, word_max_len1, -1)

        a2_word_rep = torch.FloatTensor(batch_size * word_max_len2, self.args.hidden_size * 2).zero_().to(device)
        a2_word_rep[a2_idx] = a2_rep
        a2_word_rep = a2_word_rep.view(batch_size, word_max_len2, -1)

        return a1_word_rep, a2_word_rep




    def word_encoder(self, alias, alias_word_mask, is_sample=False):
         alias_word_len = alias_word_mask.data.cpu().eq(0).long().sum(1).cpu().numpy().tolist()
         alias_rep = self.word_rnn(alias, alias_word_len, is_sample)
         return alias_rep
        
    
    def word_attn(self, a1_rep, alias1_word_mask, a2_rep, alias2_word_mask):
        a1_attn_a2 = self.word_match_attn(a1_rep, a2_rep, alias2_word_mask)
        a2_attn_a1 = self.word_match_attn(a2_rep, a1_rep, alias1_word_mask)
        a1_rep = torch.cat((a1_rep, a1_attn_a2), 2)
        a2_rep = torch.cat((a2_rep, a2_attn_a1), 2)
        a1_rep = self.word_self_attn(a1_rep, alias1_word_mask)
        a2_rep = self.word_self_attn(a2_rep, alias2_word_mask)
        return a1_rep, a2_rep
    
    
    def alias_rep(self, alias, alias_word_mask, alias_char_mask):
        char_rep = self.char_encoder(alias, alias_char_mask, is_sample=True)
        return self.word_encoder(char_rep, alias_word_mask, is_sample=True)



    def forward(self, alias1, alias1_word_mask, alias1_char_mask, alias2, alias2_word_mask, alias2_char_mask):
        a1_rep = self.char_encoder(alias1, alias1_char_mask)
        a2_rep = self.char_encoder(alias2, alias2_char_mask)

        a1_rep, a2_rep = self.char_attn(a1_rep, alias1_char_mask, a2_rep, alias2_char_mask)
        a1_rep = self.word_encoder(a1_rep, alias1_word_mask)
        a2_rep = self.word_encoder(a2_rep, alias2_word_mask)
        a1_rep, a2_rep = self.word_attn(a1_rep, alias1_word_mask, a2_rep, alias2_word_mask)

        score = self.final_layer(torch.cat((torch.abs(a1_rep - a2_rep), torch.mul(a1_rep, a2_rep)), 1))
        
        return torch.squeeze(score)
    

class Hybrid_Alias_Rep(nn.Module):
    def __init__(self, args, voc):
        super(Hybrid_Alias_Rep, self).__init__()
        self.args = args
        self.voc = voc
        self.char_rnn = Encoder_rnn(args, self.args.input_size, self.args.hidden_size)
        self.word_rnn = Encoder_rnn(args, self.args.input_size * 2, self.args.hidden_size)
        out_hidden_size = 2 * self.args.hidden_size if self.args.bidirect else self.args.hidden_size
        self.embeddings = nn.Embedding(num_embeddings=len(self.voc), embedding_dim=self.args.embedding_dim,
                                        padding_idx=0, max_norm=None, scale_grad_by_freq=False, sparse=False)
        
        self.char_self_attn = LinearAttn(out_hidden_size)
    
    
    
    def forward(self, alias1, alias1_word_mask, alias1_char_mask, alias2, alias2_word_mask, is_sample=False):
        device = torch.device("cuda")
        batch_size, word_max_len1, char_max_len1 = alias1.size(0)
        alias = alias.view(batch_size * word_max_len, char_max_len)
        
        alias_rep = self.embeddings(alias)

        alias_char_len = alias_char_mask.view(batch_size * word_max_len, char_max_len).data.eq(0).long().sum(1).cpu().numpy().tolist()
        alias_char_idx = [i for i in range(len(alias_char_len)) if alias_char_len[i] > 0]
        alias_char_mask = alias_char_mask.view(batch_size * word_max_len, char_max_len)[alias_char_idx]
        alias_char_len = [i for i in alias_char_len if i > 0]
        alias_rep = alias_rep[alias_char_idx]
        alias_rep = self.char_encoder(alias_rep, alias_char_len)




      
        alias_rep = self.char_self_attn(alias_rep, alias_char_mask)
        del alias_char_mask

        alias_word_rep = torch.FloatTensor(batch_size * word_max_len, 2 * self.args.hidden_size).zero_().to(device)
        alias_word_rep[alias_char_idx] = alias_rep
        #del alias_rep


        alias_word_rep = alias_word_rep.view(batch_size, word_max_len, -1)
        alias_word_len = alias_word_mask.data.cpu().eq(0).long().sum(1).cpu().numpy().tolist()
       
        return self.word_encoder(alias_word_rep, alias_word_len, is_sample)


class Encoder_rnn(nn.Module):
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
        #emb = self.embeddings(alias)

        

        packed_emb = torch.nn.utils.rnn.pack_padded_sequence(alias, alias_len, batch_first=True)
        output, hn = self.rnn(packed_emb)
        if self.args.self_attn and not is_sample:
            unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(output)
            unpacked = unpacked.transpose(0, 1)
            unpacked = unpacked[torch.LongTensor(unsorted_idx)]
            return unpacked
        if self.args.bidirect:
            hn = hn.view(self.args.num_layers, 2, -1, self.args.hidden_size)
            hn = hn[-1, :, :, :]
            hn = torch.cat((hn[0], hn[1]), 1) 
            
        else:
            hn = hn[-1, :, :]
                

        hn = hn[torch.LongTensor(unsorted_idx)]
        return hn


class LinearMatchAttn(nn.Module):
    def __init__(self, input_size):
        super(LinearMatchAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask, x_o):
        """
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
        output_avg = alpha.unsqueeze(1).bmm(x_o).squeeze(1)

        return output_avg


class LinearAttn(nn.Module):
    def __init__(self, input_size):
        super(LinearAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
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


class MatchAttn(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

    def __init__(self, input_size, identity=False):
        super(MatchAttn, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None



    def forward(self, x, y, y_mask):
        """
        Args:
            x: batch * len1 * hdim
            y: batch * len2 * hdim
            y_mask: batch * len2 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * hdim
        """
        # Project vectors
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y


        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1))
        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))
        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, y.size(1)), dim=-1)
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))

        # Take weighted average
        matched_seq = alpha.bmm(y)
        #residual_rep = torch.abs(x - matched_seq)

        return matched_seq



