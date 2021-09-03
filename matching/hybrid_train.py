import argparse
import torch
import dataset_hybrid
from model.rnn_encoder import Hybrid_Alias_Sim
from torch.nn import functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import numpy as np 
import time
import random
import logging
import pickle

logger = logging.getLogger()

def load_data(filename, is_lowercase):
    """
        Output:
            data: alias1(str); alias2(str), neg_alias(list[str])
    """
    data = list()
    
    for ln in open(filename, 'r').readlines():
        items = ln[:-1].split('\t') 
        if len(items) == 5:
            kb_link, alias1, alias2, neg_alias, _ = items   
        else:
            kb_link, alias1, alias2, neg_alias = items   
        if len(alias1) <= 1 or len(alias2) <= 1:
            continue
        if is_lowercase:
            alias1 = alias1.lower()
            alias2 = alias2.lower()
            neg_alias = neg_alias.lower()
        neg_alias = neg_alias.split('___')
        #if len(neg_alias) < 5: 
        #    continue
        neg = neg_alias, list()
        data.append((alias1, alias2, neg))
    return data



def load_data_train(filename, is_lowercase, pre_negscore):
    if pre_negscore is not None:
        score_ln = open(pre_negscore, 'r').readlines()
        score_dict = dict()
        for ln in score_ln:
            alias, neg_alias, neg_score = ln[:-1].split('\t')
            score_dict[alias] = {'neg':neg_alias, 'neg_score':neg_score}
    data = list()
    for ln in open(filename, 'r').readlines():
        items = ln[:-1].split('\t') 
        if len(items) == 5:
            kb_link, alias1, alias2, neg_alias, _ = items   
        else:
            kb_link, alias1, alias2, neg_alias = items   
        if len(alias1) <= 1 or len(alias2) <= 1:
            continue
        if is_lowercase:
            alias1 = alias1.lower()
            alias2 = alias2.lower()
            neg_alias = neg_alias.lower()
        if pre_negscore is not None:
            if alias1 not in score_dict:
                continue
            neg_alias = score_dict[alias1]['neg'].split('__')
            if len(neg_alias) < 20: 
                continue
            neg_score = score_dict[alias1]['neg_score'].split('__')
            neg = neg_alias, neg_score
            data.append((alias1, alias2, neg))
        else:
            neg_alias = neg_alias.split('___')
            if len(neg_alias) < 20: 
                continue
            neg = neg_alias, list()
            data.append((alias1, alias2, neg))
    return data




def load_words(exs, ngram):
    words = set()
    UNK = '<unk>'
    PAD = '<pad>'
    words.add(PAD)
    words.add(UNK)
    char2ind = {PAD: 0, UNK: 1}
    ind2char = {0: PAD, 1: UNK}
    for alias1, alias2, _ in exs:
        for i in range(0, len(alias1)-(ngram-1), ngram):
            words.add(alias1[i:i+ngram])
        if ngram == 2:
            if len(alias1) % 2 == 1:
                words.add(alias1[len(alias1)-1])
        for i in range(0, len(alias2)-(ngram-1), ngram):
            words.add(alias2[i:i+ngram])
        if ngram == 2:
            if len(alias2) % 2 == 1:
                words.add(alias2[len(alias2)-1])
    words = sorted(words)
    for w in words:
        idx = len(char2ind)
        char2ind[w] = idx
        ind2char[idx] = w
    return words, char2ind, ind2char


def train_vec(train_data, char2ind):
    train_vec = dict()
    for alias, _, neg in tqdm(train_data):
        neg_alias, _ = neg
        vec_alias1 = list()
        x1_char_len = list()
        vec_neg_alias = list()
        for word in alias.split():
            char_in_word = [char2ind[ch] if ch in char2ind else char2ind['<unk>'] for ch in word]
            vec_alias1.append(char_in_word)
            x1_char_len.append(len(word))
        
        for i, nalias in enumerate(neg_alias):
            if len(nalias) <= 1:
                continue
            vec_neg = list()
            for word in nalias.split():
                char_in_word = [char2ind[ch] if ch in char2ind else char2ind['<unk>'] for ch in word]
                vec_neg.append(char_in_word)
            if len(vec_neg) > 0:
                vec_neg_alias.append(vec_neg)
        

        x1 = torch.LongTensor(len(vec_neg_alias), len(vec_alias1), max(x1_char_len)).zero_()
        x1_word_mask = torch.ByteTensor(len(vec_neg_alias), len(vec_alias1)).fill_(1)
        x1_char_mask = torch.ByteTensor(len(vec_neg_alias), len(vec_alias1), max(x1_char_len)).fill_(1)

        for i in range(len(vec_neg_alias)):
            for j, word in enumerate(vec_alias1):
                a1 = torch.LongTensor(word)
                x1[i, j, :len(word)].copy_(a1)
                x1_char_mask[i, j, :len(word)].fill_(0)
            x1_word_mask[i, :len(vec_alias1)].fill_(0)

        x3_word_len = list()
        x3_char_len = list()
            
        for neg_alias in vec_neg_alias:
            x3_word_len.append(len(neg_alias))
            for word in neg_alias:
                x3_char_len.append(len(word))
        neg_v =  torch.LongTensor(len(x3_word_len), max(x3_word_len), max(x3_char_len)).zero_()
        neg_word_mask = torch.ByteTensor(len(x3_word_len), max(x3_word_len)).fill_(1)
        neg_char_mask = torch.ByteTensor(len(x3_word_len), max(x3_word_len), max(x3_char_len)).fill_(1)
        for i, neg_alias in enumerate(vec_neg_alias):
            for j, word in enumerate(neg_alias):
                a3 = torch.LongTensor(word)
                neg_v[i, j, :len(word)].copy_(a3)
                neg_char_mask[i, j, :len(word)].fill_(0)
            neg_word_mask[i, :len(neg_alias)].fill_(0)
        
        pos = x1, x1_word_mask, x1_char_mask
        neg = neg_v, neg_word_mask, neg_char_mask
        train_vec[alias] = {'pos':pos, 'neg':neg}
    pickle.dump(train_vec, open('train_vec.pt', 'wb'))
    return train_vec



def train_score(train_vec, model, train_data):
    device = torch.device("cuda")
    m = nn.Softmax()
    new_train_data = list()
    for alias, alias2, neg in tqdm(train_data):
        neg_alias, _ = neg
        x1, x1_word_mask, x1_char_mask = train_vec[alias]['pos']
        x1 = x1.to(device)
        x1_word_mask = x1_word_mask.to(device)
        x1_char_mask = x1_char_mask.to(device)
        neg_v, neg_word_mask, neg_char_mask = train_vec[alias]['neg']
        neg_v = neg_v.to(device)
        neg_word_mask = neg_word_mask.to(device)
        neg_char_mask = neg_char_mask.to(device)

        score = model(x1, x1_word_mask, x1_char_mask, neg_v, neg_word_mask, neg_char_mask)
        score = m(score)

        score = score.data.cpu().numpy().tolist()
        score = [str(s) for s in score]
        neg = neg_alias, score
        new_train_data.append((alias, alias2, neg))
    return new_train_data







def negative_sampling(args, model, pos_alias, pos_word_mask, pos_char_mask, neg_alias, neg_word_mask, neg_char_mask):
    pos_features = model.alias_rep(pos_alias, pos_word_mask, pos_char_mask)
    neg_features = model.alias_rep(neg_alias, neg_word_mask, neg_char_mask)
    neg_list, neg_word_len, neg_char_len = list(), list(), list()
    for i in range(len(pos_features)):
        sent1_fea = pos_features[i]
        sent1_fea = sent1_fea.repeat(neg_features.size(0), 1)
        dis = F.cosine_similarity(sent1_fea, neg_features)
        sorted, indices = torch.sort(dis, 0,  descending=True)
        indices = indices.data.cpu().numpy()
        for j in range(args.num_neg):
            ind = indices[j]
            word_len = neg_word_mask[ind].eq(0).long().sum().item()
            neg_vec = neg_alias[ind]
            neg_words = list()
            neg_word_len.append(word_len)
            for k in range(word_len):
                char_len = neg_char_mask[ind][k].eq(0).long().sum().item()
                neg_words.append(neg_alias[ind][k][:char_len])
                neg_char_len.append(char_len)
            neg_list.append(neg_words)
    
    x3 = torch.LongTensor(len(neg_word_len), max(neg_word_len), max(neg_char_len)).zero_()
    x3_word_mask = torch.ByteTensor(len(neg_word_len), max(neg_word_len)).fill_(1)
    x3_char_mask = torch.ByteTensor(len(neg_word_len), max(neg_word_len), max(neg_char_len)).fill_(1) 

    for i in range(len(neg_list)):
        vec_neg = neg_list[i]
        for j, word in enumerate(vec_neg):
            x3[i, j, :len(word)].copy_(word)
            x3_char_mask[i, j, :len(word)].fill_(0)
        x3_word_mask[i, :len(vec_neg)].fill_(0)
  
    return x3, x3_word_mask, x3_char_mask





def evaluate(args, data_loader, model, device):
    model.eval()
    ranking = 0 
    num_examples = 0
    for idx, batch in enumerate(tqdm(data_loader)):
        alias1 = batch[0].to(device)
        alias1_word_mask = batch[1].to(device)
        alias1_char_mask = batch[2].to(device)
        alias2 = batch[3].to(device)
        alias2_word_mask = batch[4].to(device)
        alias2_char_mask = batch[5].to(device)
        neg_alias_list = batch[6]
        neg_word_mask_list = batch[7]
        neg_char_mask_list = batch[8]


        pos_scores = model(alias1, alias1_word_mask, alias1_char_mask, alias2, alias2_word_mask, alias2_char_mask)
        pos_scores = pos_scores.data.cpu().numpy().tolist()
        #input()
        #####TODO
        for i in range(len(pos_scores)):
            
            neg_alias = neg_alias_list[i].to(device)
            neg_word_mask = neg_word_mask_list[i].to(device)
            neg_char_mask = neg_char_mask_list[i].to(device)

            pos_word_len = alias1_word_mask[i].data.cpu().eq(0).long().sum().item()
            pos_char_len = alias1_char_mask[i].data.cpu().eq(0).long().sum(1).numpy().tolist()

            pos_alias2 = alias1[i,:pos_word_len, :max(pos_char_len)].repeat(len(neg_alias), 1, 1)
            pos_word_mask = alias1_word_mask[i, :pos_word_len].repeat(len(neg_alias), 1)
            pos_char_mask = alias1_char_mask[i, :pos_word_len, :max(pos_char_len)].repeat(len(neg_alias), 1, 1)


            
            neg_scores = model(pos_alias2, pos_word_mask, pos_char_mask, neg_alias, neg_word_mask, neg_char_mask).data.cpu().numpy().tolist()
            pos_score = pos_scores[i]
            

            ####MRR score compute
            neg_scores.append(pos_score)
            sorted_idx = sorted(range(len(neg_scores)), key = neg_scores.__getitem__, reverse=True)
            ranking = ranking + 1/ (sorted_idx.index(len(neg_scores) - 1) + 1)
            num_examples += 1
    
    ranking /= num_examples
    logger.info("MRR SCORE IS %.5f:" % ranking)
    return ranking


def precision_recall(args, data_loader, model, device, ind2char):
    fp = open('ppl_pr_errors_1.txt', 'w')
    fp1 = open('ppl_pr_errors_0.txt', 'w')
    model.eval()
    ranking = 0 
    num_examples = 0
    recall_list = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
    pos_score_list = list()
    neg_score_list = list()

    for idx, batch in enumerate(tqdm(data_loader)):
        alias1 = batch[0].to(device)
        alias1_word_mask = batch[1].to(device)
        alias1_char_mask = batch[2].to(device)
        alias2 = batch[3].to(device)
        alias2_word_mask = batch[4].to(device)
        alias2_char_mask = batch[5].to(device)
        neg_alias_list = batch[6]
        neg_word_mask_list = batch[7]
        neg_char_mask_list = batch[8]


        pos_scores = model(alias1, alias1_word_mask, alias1_char_mask, alias2, alias2_word_mask, alias2_char_mask)
        pos_scores = pos_scores.data.cpu().numpy().tolist()

        for i in range(len(pos_scores)):
            
            neg_alias = neg_alias_list[i].to(device)
            neg_word_mask = neg_word_mask_list[i].to(device)
            neg_char_mask = neg_char_mask_list[i].to(device)

            pos_word_len = alias1_word_mask[i].data.cpu().eq(0).long().sum().item()
            pos_char_len = alias1_char_mask[i].data.cpu().eq(0).long().sum(1).numpy().tolist()

            pos_alias2 = alias1[i,:pos_word_len, :max(pos_char_len)].repeat(len(neg_alias), 1, 1)
            pos_word_mask = alias1_word_mask[i, :pos_word_len].repeat(len(neg_alias), 1)
            pos_char_mask = alias1_char_mask[i, :pos_word_len, :max(pos_char_len)].repeat(len(neg_alias), 1, 1)


            
            neg_scores = model(pos_alias2, pos_word_mask, pos_char_mask, neg_alias, neg_word_mask, neg_char_mask).data.cpu().numpy().tolist()
            pos_score = pos_scores[i]
            pos_score_list.append(pos_score)

            if pos_score < 0.247:
                ex = alias1[i].data.cpu()
                pos_ex = alias2[i].data.cpu()
                ch_list = list()

                for j, word in enumerate(ex):
                    if batch[1][i, j].item() == 1:
                        continue
                    for k, ch in enumerate(word):
                        if batch[2][i, j, k].item() == 1:
                            continue
                        else:
                            ch = ch.item()
                            if ch in ind2char:
                                ch_list.append(ind2char[ch])
                            else:
                                ch_list.append(ind2char[1])
                    ch_list.append(' ')
              

                fp1.write(''.join(ch_list[:-1]) + '\t')

                ch_list = list()
                for j, word in enumerate(pos_ex):
                    if batch[4][i, j].item() == 1:
                        continue
                    for k, ch in enumerate(word):
                        if batch[5][i, j, k].item() == 1:
                            continue
                        else:
                            ch = ch.item()
                            if ch in ind2char:
                                ch_list.append(ind2char[ch])
                            else:
                                ch_list.append(ind2char[1])
                    ch_list.append(' ')
                fp1.write(''.join(ch_list[:-1]) + '\t' + str(pos_score) + '\t' + str('1') + '\n')



            for m, v in enumerate(neg_scores):
                neg_score_list.append(v)
                if v > 6.582:
                    ex = alias1[i].data.cpu()
                    ch_list = list()

                    for j, word in enumerate(ex):
                        if batch[1][i, j].item() == 1:
                            continue
                        for k, ch in enumerate(word):
                            if batch[2][i, j, k].item() == 1:
                                continue
                            else:
                                ch = ch.item()
                                if ch in ind2char:
                                    ch_list.append(ind2char[ch])
                                else:
                                    ch_list.append(ind2char[1])
                        ch_list.append(' ')
                    
                    fp.write(''.join(ch_list[:-1]) + '\t')
                    ch_list = list()
                    
                    neg_ex = neg_alias[m].data.cpu()
                    for j, word in enumerate(neg_ex):
                        if neg_word_mask[m, j].item() == 1:
                            continue
                        for k, ch in enumerate(word):
                            if neg_char_mask[m, j, k].item() == 1:
                                continue
                            else:
                                ch = ch.item()
                                if ch in ind2char:
                                    ch_list.append(ind2char[ch])
                                else:
                                    ch_list.append(ind2char[1])
                        ch_list.append(' ')
                    fp.write(''.join(ch_list[:-1]) + '\t' + str(v) + '\t' + str('0') + '\n')






            neg_scores.append(pos_score)
            sorted_idx = sorted(range(len(neg_scores)), key = neg_scores.__getitem__, reverse=True)
            ranking = ranking + 1/ (sorted_idx.index(len(neg_scores) - 1) + 1)
            num_examples += 1
            rk = sorted_idx.index(len(neg_scores) - 1)
        
    pos_sorted_idx = sorted(range(len(pos_score_list)), key = pos_score_list.__getitem__, reverse=True)
    ranking = ranking /num_examples
    print("MRR SCORE IS: %.5f" % ranking)

    for recall in recall_list:
        num_correct_labels = int(num_examples * recall)
        score_limit = pos_score_list[pos_sorted_idx[num_correct_labels]]

        ### precision
        lb_list = [i for i in range(len(neg_score_list)) if neg_score_list[i] >= score_limit]
        precision = num_correct_labels / ( len(lb_list) + num_correct_labels)
        print('recall is: %.2f,  precision is: %.3f, score is: %.3f' %(recall, precision, score_limit))



    print('\n')
    

    
            

    


def error_analysis(args, data_loader, model, device, ind2char):
    fp = open('people_errors.txt', 'w')
    model.eval()
    ranking = 0 
    num_examples = 0
    for idx, batch in enumerate(tqdm(data_loader)):
        alias1 = batch[0].to(device)
        alias1_word_mask = batch[1].to(device)
        alias1_char_mask = batch[2].to(device)
        alias2 = batch[3].to(device)
        alias2_word_mask = batch[4].to(device)
        alias2_char_mask = batch[5].to(device)
        neg_alias_list = batch[6]
        neg_word_mask_list = batch[7]
        neg_char_mask_list = batch[8]


        pos_scores = model(alias1, alias1_word_mask, alias1_char_mask, alias2, alias2_word_mask, alias2_char_mask)
        pos_scores = pos_scores.data.cpu().numpy().tolist()
        correct_ex = 0
  
        for i in range(len(pos_scores)):
            
            neg_alias = neg_alias_list[i].to(device)
            neg_word_mask = neg_word_mask_list[i].to(device)
            neg_char_mask = neg_char_mask_list[i].to(device)

            pos_word_len = alias1_word_mask[i].data.cpu().eq(0).long().sum().item()
            pos_char_len = alias1_char_mask[i].data.cpu().eq(0).long().sum(1).numpy().tolist()

            pos_alias2 = alias1[i,:pos_word_len, :max(pos_char_len)].repeat(len(neg_alias), 1, 1)
            pos_word_mask = alias1_word_mask[i, :pos_word_len].repeat(len(neg_alias), 1)
            pos_char_mask = alias1_char_mask[i, :pos_word_len, :max(pos_char_len)].repeat(len(neg_alias), 1, 1)


            
            neg_scores = model(pos_alias2, pos_word_mask, pos_char_mask, neg_alias, neg_word_mask, neg_char_mask).data.cpu().numpy().tolist()
            pos_score = pos_scores[i]
            

            ####MRR score compute
            neg_scores.append(pos_score)
            sorted_idx = sorted(range(len(neg_scores)), key = neg_scores.__getitem__, reverse=True)
            ranking = ranking + 1/ (sorted_idx.index(len(neg_scores) - 1) + 1)
            num_examples += 1
            rk = sorted_idx.index(len(neg_scores) - 1)
            if rk == 0 :
                correct_ex += 1
            if rk > 0:
                ex = alias1[i].data.cpu()
                pos_ex = alias2[i].data.cpu()
                pos_sc = pos_score

                #sec_rk = sorted_index[1]
                neg_ex = neg_alias[sorted_idx[0]].data.cpu()
                neg_sc = neg_scores[sorted_idx[0]]
                ch_list = list()

                for j, word in enumerate(ex):
                    if batch[1][i, j].item() == 1:
                        continue
                    for k, ch in enumerate(word):
                        if batch[2][i, j, k].item() == 1:
                            continue
                        else:
                            ch = ch.item()
                            if ch in ind2char:
                                ch_list.append(ind2char[ch])
                            else:
                                ch_list.append(ind2char[1])
                    ch_list.append(' ')
              

                fp.write(''.join(ch_list[:-1]) + '\t')
                ch_list = list()
                for j, word in enumerate(pos_ex):
                    if batch[4][i, j].item() == 1:
                        continue
                    for k, ch in enumerate(word):
                        if batch[5][i, j, k].item() == 1:
                            continue
                        else:
                            ch = ch.item()
                            if ch in ind2char:
                                ch_list.append(ind2char[ch])
                            else:
                                ch_list.append(ind2char[1])
                    ch_list.append(' ')
                fp.write(''.join(ch_list[:-1]) + '\t' + str(pos_sc) + '\t')
                ch_list = list()
                for j, word in enumerate(neg_ex):
                    if neg_word_mask[sorted_idx[0], j].item() == 1:
                        continue
                    for k, ch in enumerate(word):
                        if neg_char_mask[sorted_idx[0], j, k].item() == 1:
                            continue
                        else:
                            ch = ch.item()
                            if ch in ind2char:
                                ch_list.append(ind2char[ch])
                            else:
                                ch_list.append(ind2char[1])
                    ch_list.append(' ')
                fp.write(''.join(ch_list[:-1]) + '\t' + str(neg_sc) + '\n')

        

           
    
    ranking /= num_examples
    correct_ex /= num_examples
    logger.info("MRR SCORE IS: %.5f" % ranking)
    logger.info('p@1 is %.5f' % correct_ex)
    return ranking





def train(args, data_loader, val_train_loader, model, device, best_mrr):
    model.train()
    optimizer = torch.optim.Adamax(model.parameters())
    print_loss_total = 0 
    epoch_loss_total = 0
    start = time.time()
    check_point = 100

    for idx, batch in enumerate(tqdm(data_loader)):
        alias1 = batch[0].to(device)
        alias1_word_mask = batch[1].to(device)
        alias1_char_mask = batch[2].to(device)
        alias2 = batch[3].to(device)
        alias2_word_mask = batch[4].to(device)
        alias2_char_mask = batch[5].to(device)
        #neg_alias = batch[6].to(device)
        #neg_word_mask = batch[7].to(device)
        #neg_char_mask = batch[8].to(device)
        pos_score = model(alias1, alias1_word_mask, alias1_char_mask, alias2, alias2_word_mask, alias2_char_mask)
        pos_score = pos_score.sigmoid().log().sum()
        loss = pos_score
        

        for i in range(args.num_neg):
            neg_alias = batch[6][i].to(device)
            neg_word_mask = batch[7][i].to(device)
            neg_char_mask = batch[8][i].to(device)
            neg_score = model(alias1, alias1_word_mask, alias1_char_mask, neg_alias, neg_word_mask, neg_char_mask)
            neg_score = neg_score.neg().sigmoid().log().sum()
            loss = loss + neg_score
            

    
        ### negative sample selection
        #neg_alias, neg_word_mask, neg_char_mask = negative_sampling(args, model, alias1, alias1_word_mask, alias1_char_mask, neg_alias, neg_word_mask, neg_char_mask)
        
        #pos2_word_mask = alias1_word_mask.repeat(1,args.num_neg).view(-1, alias1_word_mask.size(1)).to(device)
        #pos2_char_mask = alias1_char_mask.view(-1, alias1_char_mask.size(1) * alias1_char_mask.size(2)).repeat(1,args.num_neg).view(-1, alias1_char_mask.size(1), alias1_char_mask.size(2)).to(device)
        #pos_alias2 = alias1.view(-1, alias1.size(1) * alias1.size(2)).repeat(1, args.num_neg).view(-1, alias1.size(1), alias1.size(2)).to(device)


        #neg_alias = neg_alias.to(device)
        #neg_word_mask = neg_word_mask.to(device)
        #neg_char_mask = neg_char_mask.to(device)
        #### tile operation

        #neg_score = model(pos_alias2, pos2_word_mask, pos2_char_mask, neg_alias, neg_word_mask, neg_char_mask)
        #neg_score = neg_score.neg().sigmoid().log().sum()


        #neg_score = model(pos_alias2, pos2_word_mask, pos2_char_mask, neg_alias, neg_word_mask, neg_char_mask)
        #neg_score = neg_score.neg().sigmoid().log().sum()



        #loss = pos_score + neg_score
        loss = -loss.sum() / alias1.size(0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        clip_grad_norm_(model.parameters(), 5)
        print_loss_total += loss.data.cpu().numpy()
        epoch_loss_total += loss.data.cpu().numpy()
        
        if idx % check_point ==0 and idx > 0:
            print_loss_total = print_loss_total
            print_loss_avg = print_loss_total / check_point
                
            logger.info('number of steps: %d, loss: %.5f time: %.5f' % (idx, print_loss_avg, time.time()- start))
            print_loss_total = 0
            mrr_score = evaluate(args, val_train_loader, model, device)    

            if mrr_score > best_mrr:
                torch.save(model, args.save_model)
                best_mrr = mrr_score
            model.train()
            torch.cuda.empty_cache()

    logger.info('epoch loss is: %.5f' % epoch_loss_total)
    return best_mrr









def main():
    parser = argparse.ArgumentParser(description='Alias Similarity')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--data-workers', type=int, default=2)
    parser.add_argument('--train-file', type=str, default='../data/your_data.txt')
    parser.add_argument('--dev-file', type=str, default='../data/your_data.txt')
    parser.add_argument('--test-file', type=str, default='../data/your_data.txt')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--input-size', type=int, default=300)
    parser.add_argument('--hidden-size', type=int, default=300)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=int, default=0.4)
    parser.add_argument('--embedding-dim', type=int, default=300)
    parser.add_argument('--bidirect', action='store_true', default=True)
    parser.add_argument('--num-neg', type=int, default=5)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--n-gram', type=int, default=1)
    parser.add_argument('--transfer', action='store_true', default=False)
    parser.add_argument('--base-model', type=str, default='../model/model.pt')
    parser.add_argument('--save-model', type=str, default='../model/hybrid.pt')
    parser.add_argument('--load-model', type=str, default='../model/hybrid.pt')
    parser.add_argument('--lowercase', action='store_true', default=False)
    parser.add_argument('--self-attn', action='store_true', default=True)
    parser.add_argument('--log-file', type=str, default = '../log/log_file.log')
    parser.add_argument('--pre-negscore', type=str, default=None)


    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logfile = logging.FileHandler(args.log_file, 'a')

    logfile.setFormatter(fmt)
    logger.addHandler(logfile)

    train_exs = load_data_train(args.train_file, args.lowercase, args.pre_negscore)
    dev_exs = load_data(args.dev_file, args.lowercase)
    #vocab_dict = voc, char2ind, ind2char
    #pickle.dump(vocab_dict, open('../trained_model/vocab.pkl', 'wb'))
    #exit()

    if args.transfer:
        logger.info('transfer learning')
        voc, char2ind, ind2char = pickle.load(open('../trained_model/vocab.pkl', 'rb'))
        model = torch.load(args.base_model)
    else:
        voc, char2ind, ind2char = load_words(train_exs + dev_exs, args.n_gram)
        vocab_dict = voc, char2ind, ind2char
        #pickle.dump(vocab_dict, open('../trained_model/pre_trained/vocab_movie.pkl', 'wb'))
        #exit()
        model = Hybrid_Alias_Sim(args, voc)


    if args.resume:
        logger.info('use previous model')
        model = torch.load(args.load_model)
    
    device = torch.device("cuda" if args.cuda else "cpu")
    model.to(device)
    #logger.info(model)

    if args.test:
        test_exs = load_data(args.test_file, args.lowercase)
        test_dataset = dataset_hybrid.AliasDataset(test_exs, ind2char, voc, char2ind, args.n_gram, args.num_neg)
        test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                               sampler=test_sampler, num_workers=args.data_workers,
                                               collate_fn=dataset_hybrid.val_batchify, pin_memory=args.cuda)
        precision_recall(args, test_loader, model, device, ind2char)
        #error_analysis(args, test_loader, model, device, ind2char)

        exit()
    
    #train_vec_rep = train_vec(train_exs, char2ind)
    #train_vec_rep = pickle.load(open('train_vec.pt', 'rb'))


    train_dataset = dataset_hybrid.AliasDataset(train_exs, ind2char, voc, char2ind, args.n_gram, args.num_neg)
    train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)

    dev_dataset = dataset_hybrid.AliasDataset(dev_exs[:1000], ind2char, voc, char2ind, args.n_gram, args.num_neg)
    dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size,
                                               sampler=dev_sampler, num_workers=args.data_workers,
                                               collate_fn=dataset_hybrid.val_batchify, pin_memory=args.cuda)
   
    start_epoch = 0
    logger.info('start training:')
    best_mrr = 0
    
    for epoch in range(start_epoch, args.num_epochs):
        logger.info('start epoch:%d' % epoch)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               sampler=train_sampler, num_workers=args.data_workers,
                                               collate_fn=dataset_hybrid.train_batchify, pin_memory=args.cuda)
        best_mrr = train(args, train_loader, dev_loader, model, device, best_mrr)






if __name__ == "__main__":
    main()
