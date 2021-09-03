import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import random
import torch.nn as nn




def table_vectorize(ex, char2ind):
    #### vectorize each table column, we choose only first 10 distinct values

    kb_link, alias_list, alias_type = ex
    vec_alias1 = list()
    
    for i, alias in enumerate(alias_list.split('___')[:10]):
        vec_alias = list()
        for word in alias.split():
            char_in_word = [char2ind[ch] if ch in char2ind else char2ind['<unk>'] for ch in word]
            vec_alias.append(char_in_word)
        if len(vec_alias) > 0:
            vec_alias1.append(vec_alias)
    
    return vec_alias1, alias_type, alias_list, kb_link



class Table_Alias_Dataset(Dataset):
    def __init__(self, examples, ind2char, voc, char2ind):
        self.examples = examples
        self.ind2char = ind2char
        self.voc = voc
        self.char2ind = char2ind
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        return table_vectorize(self.examples[index], self.char2ind)
    
    def lengths(self):
        return [len(alias1) for _, alias1 in self.examples]


def table_batchify(batch):
    #### batchify the vectors

    x1, x1_word_mask, x1_char_mask = list(), list(), list()
    alias_str, kb_list, header_list = list(), list(), list()
    for ex in batch:
        x_word_len, x_char_len = list(), list()
        for alias in ex[0]:
            x_word_len.append(len(alias))
            for word in alias:
                x_char_len.append(len(word))
        neg_v =  torch.LongTensor(len(x_word_len), max(x_word_len), max(x_char_len)).zero_()
        neg_word_mask = torch.ByteTensor(len(x_word_len), max(x_word_len)).fill_(1)
        neg_char_mask = torch.ByteTensor(len(x_word_len), max(x_word_len), max(x_char_len)).fill_(1)
        for i, alias in enumerate(ex[0]):
            for j, word in enumerate(alias):
                a3 = torch.LongTensor(word)
                neg_v[i, j, :len(word)].copy_(a3)
                neg_char_mask[i, j, :len(word)].fill_(0)
            neg_word_mask[i, :len(alias)].fill_(0)
        x1.append(neg_v)
        x1_word_mask.append(neg_word_mask)
        x1_char_mask.append(neg_char_mask)
        alias_str.append(ex[2])
        kb_list.append(ex[3])
        header_list.append(ex[1])
        
    return x1, x1_word_mask, x1_char_mask, alias_str, kb_list, header_list



def vectorize(ex, char2ind):
    #### vectorize each example

    alias1, alias_type = ex
    vec_alias1 = list()
    for word in alias1.split():
        char_in_word = [char2ind[ch] if ch in char2ind else char2ind['<unk>'] for ch in word]
        vec_alias1.append(char_in_word)
    return vec_alias1, alias_type




class AliasDataset(Dataset):
    def __init__(self, examples, ind2char, voc, char2ind, ngram):
        self.examples = examples
        self.ind2char = ind2char
        self.voc = voc
        self.char2ind = char2ind
        self.ngram = ngram
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        return vectorize(self.examples[index], self.char2ind)
    
    def lengths(self):
        return [len(alias1) for alias1, _ in self.examples]




def transfer_batchify(batch):
    ### batchify for transfer learning, here we have only one output(transfered domain)

    x1_word_len, x1_char_len = list(), list()
    target_label1 = list()

    for ex in batch:
        vec_alias1 = ex[0]
        x1_word_len.append(len(vec_alias1))
        for word in vec_alias1:
            x1_char_len.append(len(word))
        target_label1.append(int(ex[1]))
    
    target_label1 = torch.LongTensor(target_label1)

    x1 = torch.LongTensor(len(x1_word_len), max(x1_word_len), max(x1_char_len)).zero_()
    x1_word_mask = torch.ByteTensor(len(x1_word_len), max(x1_word_len)).fill_(1)
    x1_char_mask = torch.ByteTensor(len(x1_word_len), max(x1_word_len), max(x1_char_len)).fill_(1)

    for i in range(len(x1_word_len)):
        vec_alias1 = batch[i][0]

        for j, word in enumerate(vec_alias1):
            a1 = torch.LongTensor(word)
            x1[i, j, :len(word)].copy_(a1)
            x1_char_mask[i, j, :len(word)].fill_(0)
        x1_word_mask[i, :len(vec_alias1)].fill_(0)

      
    return x1, x1_word_mask, x1_char_mask, target_label1






def batchify(batch):
    #### batchify the vectors
    '''
    input : batches
    output: 
        Alias(x1) : num_examples * max_word_len * max_char_len
        Word_mask(x1_word_mask) : num_examples * max_word_len
        Char_mask(x1_char_mask) : num_examples * max_word_len * max_char_len
        Label : num of lists, each with num of examples length

    '''
    num_labels = len(batch[0][1].split('___'))
    x1_word_len, x1_char_len = list(), list()
    target_label_list = list()
    for i in range(53):
        target_label_list.append(list())

    for ex in batch:
        vec_alias1 = ex[0]
        x1_word_len.append(len(vec_alias1))
        for word in vec_alias1:
            x1_char_len.append(len(word))
        label_list = ex[1].split('___')
        for i in range(len(label_list)):
            target_label_list[i].append(int(label_list[i]))
    
    x1 = torch.LongTensor(len(x1_word_len), max(x1_word_len), max(x1_char_len)).zero_()
    x1_word_mask = torch.ByteTensor(len(x1_word_len), max(x1_word_len)).fill_(1)
    x1_char_mask = torch.ByteTensor(len(x1_word_len), max(x1_word_len), max(x1_char_len)).fill_(1)

    for i in range(len(x1_word_len)):
        vec_alias1 = batch[i][0]
        for j, word in enumerate(vec_alias1):
            a1 = torch.LongTensor(word)
            x1[i, j, :len(word)].copy_(a1)
            x1_char_mask[i, j, :len(word)].fill_(0)
        x1_word_mask[i, :len(vec_alias1)].fill_(0)

    return x1, x1_word_mask, x1_char_mask, target_label_list
    