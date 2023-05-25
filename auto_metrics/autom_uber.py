import os
import sys
sys.path.append(os.getcwd())
import torch 
import numpy as np 
import pandas as pd
from dataclasses import dataclass, field
import json
import logging
import os
from typing import Optional
import argparse

from hf_metrics_utils import *
from fast_bleu import SelfBLEU

import _pickle as cPickle
import pickle


def read_pickle(filepath, filename):

        f = open(os.path.join(filepath, filename), 'rb')
        read_file = cPickle.load(f)
        f.close()

        return read_file

def save_pickle(filepath, filename, data):

    f = open(os.path.join(filepath, filename), 'wb')
    cPickle.dump(data, f)
    print(" file saved to: %s"%(os.path.join(filepath, filename)))
    f.close()



def parse_args():
    parser = argparse.ArgumentParser(description="Training a multi-modal multi-label classification")

    parser.add_argument("--attribute", type=str, help="senti vs. topic as control attribute.")
    parser.add_argument("--domain", type=str, default='rocstories', help="Type of evaluator: rocstories vs. writingprompts.")
    parser.add_argument("--task", type=str, default='binary', help="Classification task: binary vs. multilabel.")
    parser.add_argument('-m', '--metric', nargs='+', default=[], help="Metric as feature.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--nlabels", type=int, default=None, help="Number of multi-labels.")
    args = parser.parse_args()


    return args



def read_data(args):

  datadir = args.output_dir
  
  # based on pairwise human ratings, the order of model pair is important
  files = ["uber_%.csv"%(args.attribute), \
            ]

  print(datadir)

  data_df = pd.read_csv(os.path.join(datadir, files[0]))
  
  
  return data_df

def compute_perplexity_df(data):

    texts = data['text_clean'].values
    scores = np.array(compute_perplexity(texts, model='gpt2'))
    data['perplexity'] = scores

    return data


def compute_selfbleu_df(data):

    texts = data['text_clean'].values
    tokenized_texts = [t.split() for t in texts]
    weights = {'bigram': (1/2., 1/2.), 'trigram': (1/3., 1/3., 1/3.)}
    self_bleu = SelfBLEU(tokenized_texts, weights)

    results = self_bleu.get_score()
    results_2 = np.array(results['bigram'])
    results_3 = np.array(results['trigram'])

    data['self_bleu2'] = results_2
    data['self_bleu3'] = results_3    

    return data



def read_keywords(args):

    if args.attribute == 'topic':
        files = ['computers.txt', \
            'legal.txt', \
            'military.txt', \
            'politics.txt', \
            'religion.txt', \
            'science.txt', \
            'space.txt']
    else:
        files = ['negative.txt', 'positive.txt']

    keyword_dict = {}
    for f in files:
        df = pd.read_csv(os.path.join("data/keywords", f), header=None, names=['keywords'])
        kws = df.keywords.values
        nm = os.path.splitext(f)[0]
        if args.attribute == 'topic':
            keyword_dict[nm] = ' '.join(kws)
        else:
            keyword_dict[nm] = ' '.join(kws)

    print(keyword_dict.keys())

    return keyword_dict

def compute_bertscore_df(args, data):

    # get keywords 

    keyword_dict = read_keywords(args)

    texts = data['text_clean'].values
    attribute = data['domain'].values

    print(set(attribute))

    list_kws = []
    for att in attribute:
        list_kws.append(keyword_dict[str(att)])

    results_precis, results_recall, results_f1 = compute_bertscore(texts, list_kws)

    data['bert_f1'] = np.array(results_precis)
    data['bert_precision'] = np.array(results_recall)
    data['bert_recall'] = np.array(results_f1)


    return data
if __name__ == "__main__":
    args = parse_args()

    # read data

    data_df = read_data(args)

    # compute perplexity 
    print("Computing Perplexity...")
    new_df = compute_perplexity_df(data_df)
    
    # compute BLEU

    print("Computing Self BLEU...")
    new_df = compute_selfbleu_df(new_df)

    
    
    # computing BERTScore 

    print("Computing BERTScore for all data...")
    new_datadf = compute_bertscore_df(args, new_df)

    outdir = os.path.join(args.output_dir, 'uber_%s_auto.csv'%(args.attribute))
    new_datadf.to_csv(outdir)

  
    
