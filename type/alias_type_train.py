import argparse
import torch
import alias_type_dataset
from type_model import Alias_Type
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

def isfloat(value):
  try:
    float(value)
    return True
  except:
    return False

def CheckAsciichar(input_string):
    for ch in input_string:
        if ord(ch) >= 128:
            return False
    return True


def load_table_data(filename, is_lowercase):
    data = list()
    for ln in tqdm(open(filename, 'r').readlines()):
        kb_link, _, _, _, alias_type, aliases = ln[:-1].split('\t') 
        alias_list = aliases.split('___')
        flitered = False
        for als in alias_list:
            if isfloat(als):
                flitered = True
                break
            if not CheckAsciichar(als):
                flitered = True
                break
            if len(als) < 3:
                flitered = True
                break
        if flitered:
            continue
        if is_lowercase:
           aliases = aliases.lower()
        data.append((kb_link, aliases, alias_type))
    return data



def load_data(filename, is_lowercase):
    """
        Output:
            data: alias1(str); alias2(str), neg_alias(list[str])
    """

    data = list()
    for ln in open(filename, 'r').readlines():
        kb_link, alias1, alias_type = ln[:-1].split('\t')   
        if len(alias1) <= 1:
            continue
        if is_lowercase:
            alias1 = alias1.lower()
        data.append((alias1, alias_type))
    return data



def load_words(exs, ngram):
    words = set()
    UNK = '<unk>'
    PAD = '<pad>'
    words.add(PAD)
    words.add(UNK)
    char2ind = {PAD: 0, UNK: 1}
    ind2char = {0: PAD, 1: UNK}
    for alias1, _ in exs:
        for i in range(0, len(alias1)-(ngram-1), ngram):
            words.add(alias1[i:i+ngram])
        if ngram == 2:
            if len(alias1) % 2 == 1:
                words.add(alias1[len(alias1)-1])
    words = sorted(words)
    for w in words:
        idx = len(char2ind)
        char2ind[w] = idx
        ind2char[idx] = w
    return words, char2ind, ind2char



def write_line(fp, alias, alias_word_mask, alias_char_mask, label, predict, predict_score, category, ind2char):
    ch_list = list()
    for j, word in enumerate(alias):
        if alias_word_mask[j].item() == 1:
            continue
        for k, ch in enumerate(word):
            if alias_char_mask[j, k].item() == 1:
                continue
            ch = ch.item()
            if ch in ind2char:
                ch_list.append(ind2char[ch])
            else:
                ch_list.append(ind2char[1])
        ch_list.append(' ')
    fp.write(''.join(ch_list[:-1]) + '\t' + str(label) + '\t' + str(predict) + '\t' + str(predict_score[0]) + '\t' + str(predict_score[1]) + '\t'  + str(category) + '\n')



def table_analysis(args, data_loader, model, device, ind2char):
    fp = open('table_result2.txt', 'w')
    model.eval()
    types = ['building', 'astronomical_discoverery', 'airport', 'award', 'protein', \
            'written_work', 'business.employer', 'chemical_compound', 'product', 'computer', 'software', 'conference', \
            'educational_institute', 'film', 'food', 'human_language', 'local.entity', 'location', 'populated_place', \
            'album', 'organization', 'person', 'real_estate.house', 'sports.team', 'sports.facility', 'sports.game', \
            'time.recurring_event', 'architechure.venue', 'geographical_feature', 'gene', 'job_title', 'videogame', 'academic_post_title', \
            'aviation.airline', 'miliitary.unit', 'location.city', 'book.scholar_work', 'auto.model', 'auto.model_year', 'auto.trim_level', \
            'addr_au', 'addr_ca', 'addr_gb', 'addr_ie', 'addr_in', 'addr_nz', 'addr_ph', 'add_us', 'addr_za', 'date', 'height', 'weight', \
            'longitute/latitute']
    fp.write('\t\t\t')
    for tp in types:
        fp.write(tp + '\t' + '000' + '\t' + '000' + '\t')
    fp.write('\n')

    for idx, batch in enumerate(tqdm(data_loader)):
        num_examples = len(batch[0])
        for i in range(num_examples):
            alias1 = batch[0][i].to(device)
            alias1_word_mask = batch[1][i].to(device)
            alias1_char_mask = batch[2][i].to(device)
            logit_list = model(alias1, alias1_word_mask, alias1_char_mask)
            out_str = batch[3][i] + '\t' + batch[4][i] + '\t' + batch[5][i] + '\t'
            label_ct = 0 
            for j, logit in enumerate(logit_list):
                top_n, top_i =torch.mean(logit, 0).topk(1)
                top_i = top_i.data.cpu().item()
                logit = torch.mean(logit, 0).data.cpu().numpy().tolist()
                out_str = out_str + str(top_i) + '\t' + str(logit[0]) + '\t' + str(logit[1]) + '\t' 
                label_ct += top_i
            if label_ct == 0:
                continue
            out_str += '\n'
            fp.write(out_str)



def precision_recall(args, data_loader, model, device, ind2char):
    model.eval()
    score_list = dict()
    all_labels = dict()
    for i in range(args.num_classes):
        score_list[i] = list()
        all_labels[i] = list()
    
    candidate_set = [3, 4, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, \
                    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, \
                    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]

    recall_list = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5]

    for idx, batch in enumerate(tqdm(data_loader)):
        alias1 = batch[0].to(device)
        alias1_word_mask = batch[1].to(device)
        alias1_char_mask = batch[2].to(device)
        label_list = batch[3]
        logit_list = model(alias1, alias1_word_mask, alias1_char_mask)



        for jj in range(alias1.size(0)):
            for i in range(args.num_classes):
                score = logit_list[i][jj, 1].data.cpu().item()
                label = label_list[i][jj]
                if label_list[i][jj] == -1:
                    continue

                score_list[i].append(score)
                all_labels[i].append(label)
                
    for i in range(args.num_classes):
        scores = score_list[i]
        labels = all_labels[i]
        ranked_idx = sorted(range(len(scores)), key = scores.__getitem__, reverse=True)

        for recall in recall_list:
            label_sum = 0

            num_correct_labels = int(sum(labels) * recall)
            for idx in ranked_idx:
                label_sum += labels[idx]
                if label_sum == num_correct_labels:
                    score_limit = scores[idx]
                    break
            ### precision
            lb_list = [labels[j] for j, score in enumerate(scores) if score >= score_limit]
            precision = sum(lb_list) / len(lb_list)
            print(candidate_set[i], 'recall is: ', recall, ' precision is: ', precision, 'score is :', score_limit)
        print('\n')

        
                
        




def evaluate(args, data_loader, model, device):
    model.eval()
    num_examples = [0] * args.num_classes
    error_list = [0] * args.num_classes

    for idx, batch in enumerate(tqdm(data_loader)):
        alias1 = batch[0].to(device)
        alias1_word_mask = batch[1].to(device)
        alias1_char_mask = batch[2].to(device)
        label_list = batch[3]


        logit_list = model(alias1, alias1_word_mask, alias1_char_mask)
        for i in range(args.num_classes):
            top_n, top_i = logit_list[i].topk(1)
            labels = label_list[i]
            remained_idx = [j for j in range(len(labels)) if labels[j] != -1]
            error_list[i] += torch.abs(torch.sum(top_i[remained_idx].squeeze().cpu() - torch.LongTensor(label_list[i])[remained_idx])).item()
            num_examples[i] += len(remained_idx)

        
    error_sum = 0
    for i in range(args.num_classes):
        error = error_list[i] / num_examples[i]
        print(('error is', i, error, error_list[i]))
        error_sum += error


    total_error = error_sum / args.num_classes
    print('total error', total_error)
    return total_error




def train(args, data_loader, val_train_loader, model, device, min_error):
    model.train()
    optimizer = torch.optim.Adamax(model.parameters())
    print_loss_total = 0 
    epoch_loss_total = 0
    start = time.time()
    criterion = nn.CrossEntropyLoss()


    for idx, batch in enumerate(tqdm(data_loader)):
        alias1 = batch[0].to(device)
        alias1_word_mask = batch[1].to(device)
        alias1_char_mask = batch[2].to(device)
        label_list = batch[3]
        

        logit_list = model(alias1, alias1_word_mask, alias1_char_mask)
        tgt_label = torch.LongTensor(label_list[0]).to(device)

        loss = 0
        for i in range(len(label_list)):
            labels = label_list[i]
            #### only update with non-conflict lables
            remained_idx = [j for j in range(len(labels)) if labels[j] != -1]

            logits = logit_list[i][remained_idx]

            loss += criterion(logits, torch.LongTensor(label_list[i])[remained_idx].to(device))
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        clip_grad_norm_(model.parameters(), 5)
        print_loss_total += loss.data.cpu().numpy()
        epoch_loss_total += loss.data.cpu().numpy()
        
        if idx % 600 ==0 and idx > 0:
            print_loss_total = print_loss_total
            print_loss_avg = print_loss_total / 600
                
            logger.info('number of steps: %d, loss: %.5f time: %.5f' % (idx, print_loss_avg, time.time()- start))
            print_loss_total = 0
            error = evaluate(args, val_train_loader, model, device)    

            if error < min_error:
                torch.save(model, args.save_model)
                min_error = error
            model.train()
            torch.cuda.empty_cache()

    logger.info('epoch loss is: %.5f' % epoch_loss_total)
    return min_error



def main():
    parser = argparse.ArgumentParser(description='Alias Type')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--data-workers', type=int, default=0)
    parser.add_argument('--train-file', type=str, default='../data/your_name.txt')
    parser.add_argument('--dev-file', type=str, default='../data/your_name.txt')
    parser.add_argument('--test-file', type=str, default='../data/your_name.txt')
    parser.add_argument('--table-file', type=str, default='../data/your_name.txt')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--input-size', type=int, default=300)
    parser.add_argument('--hidden-size', type=int, default=300)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=int, default=0.4)
    parser.add_argument('--embedding-dim', type=int, default=300)
    parser.add_argument('--num-classes', type=int, default=53)
    parser.add_argument('--bidirect', action='store_true', default=True)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--table-test', action='store_true', default=False)
    parser.add_argument('--n-gram', type=int, default=1)
    parser.add_argument('--save-model', type=str, default='../model/type_all.pt')
    parser.add_argument('--load-model', type=str, default='../model/type_all.pt')
    parser.add_argument('--lowercase', action='store_true', default=False)
    parser.add_argument('--log-file', type=str, default = '../log/type_log_file.log')
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

    #device = torch.device("cuda" if args.cuda else "cpu")
    device = torch.device("cpu")

    if args.table_test:
        ##### Run table test
        model = torch.load(args.load_model)
        model.to(device)
        voc = model.voc
        ind2char = model.ind2char
        char2ind = model.char2ind
        table_exs = load_table_data(args.table_file, args.lowercase)
        table_dataset = alias_type_dataset.Table_Alias_Dataset(table_exs, ind2char, voc, char2ind)
        table_sampler = torch.utils.data.sampler.SequentialSampler(table_dataset)
        table_loader = torch.utils.data.DataLoader(table_dataset, batch_size=args.batch_size,
                                               sampler=table_sampler, num_workers=args.data_workers,
                                               collate_fn=alias_type_dataset.table_batchify, pin_memory=args.cuda)
        table_analysis(args, table_loader, model, device, ind2char)

    elif args.test:
        ##### Run test
        print('test')
        model = torch.load(args.load_model)
        model.to(device)
        voc = model.voc
        ind2char = model.ind2char
        char2ind = model.char2ind
        test_exs = load_data(args.test_file, args.lowercase)
        test_dataset = alias_type_dataset.AliasDataset(test_exs, ind2char, voc, char2ind, args.n_gram)
        test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                               sampler=test_sampler, num_workers=args.data_workers,
                                               collate_fn=alias_type_dataset.batchify, pin_memory=args.cuda)
                                               
        ### Output the P/R curve for each class
        #error_analysis(args, test_loader, model, device, ind2char)
        precision_recall(args, test_loader, model, device, ind2char)
    else:
        #### Training
        if args.resume:
            logger.info('use previous model')
            model = torch.load(args.load_model)
            model.to(device)
            voc = model.voc
            ind2char = model.ind2char
            char2ind = model.char2ind
        else:
            train_exs = load_data(args.train_file, args.lowercase)
            dev_exs = load_data(args.dev_file, args.lowercase)
            voc, char2ind, ind2char = load_words(train_exs + dev_exs, args.n_gram)
            model = Alias_Type(args, voc, ind2char, char2ind)
            model.to(device)


        train_dataset = alias_type_dataset.AliasDataset(train_exs, ind2char, voc, char2ind, args.n_gram)
        train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)

        dev_dataset = alias_type_dataset.AliasDataset(dev_exs, ind2char, voc, char2ind, args.n_gram)
        dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
        dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size,
                                               sampler=dev_sampler, num_workers=args.data_workers,
                                               collate_fn=alias_type_dataset.batchify, pin_memory=args.cuda)
   
        logger.info('start training:')
        min_error = 1e4
        for epoch in range(args.num_epochs):
            logger.info('start epoch:%d' % epoch)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               sampler=train_sampler, num_workers=args.data_workers,
                                               collate_fn=alias_type_dataset.batchify, pin_memory=args.cuda)
            min_error = train(args, train_loader, dev_loader, model, device, min_error)






if __name__ == "__main__":
    main()
