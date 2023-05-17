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
    parser.add_argument("--task", type=str, default='binary', help="Classification task: binary vs. multilabel.")
    parser.add_argument('-m', '--metric', nargs='+', default=[], help="Metric as feature.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--nlabels", type=int, default=None, help="Number of multi-labels.")
    args = parser.parse_args()


    return args



def read_data(args):

  datadir = args.output_dir
  #file = 'ctrl_diag.csv'
  files = ['ctrlEval_%.csv'%(args.attribute), \
            ]
  
  print(datadir)
  print(files)

  data_df = pd.read_csv(os.path.join(datadir, files[0]))
  
  
  return data_df


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
            '''
            # for CTRL data
            if nm == 'computers':
                keyword_dict['technologies'] = ' '.join(kws)
            else:
                keyword_dict[nm] = ' '.join(kws)
            '''
        else:
            keyword_dict[nm] = ' '.join(kws)

    print(keyword_dict.keys())

    return keyword_dict

def compute_bertscore_df(data):

    # get keywords 

    keyword_dict = read_keywords(args)

    texts = data['output'].values
    attribute = data['attribute'].values

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
    

    # computing BERTScore 

    print("Computing BERTScore for all data...")
    new_datadf = compute_bertscore_df(data_df)

    outdir = os.path.join(args.output_dir, 'ctrlEval_%s_auto.csv'%(args.attribute))
    new_datadf.to_csv(outdir)

    