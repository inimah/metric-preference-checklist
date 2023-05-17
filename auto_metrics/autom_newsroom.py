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


    parser.add_argument("--domain", type=str, default='rocstories', help="Type of evaluator: rocstories vs. writingprompts.")
    parser.add_argument("--task", type=str, default='binary', help="Classification task: binary vs. multilabel.")
    parser.add_argument('-m', '--metric', nargs='+', default=[], help="Metric as feature.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--nlabels", type=int, default=None, help="Number of multi-labels.")
    args = parser.parse_args()


    return args



def read_data(args):

  datadir = args.output_dir
  #file = 'newsroom.csv'
  #file = 'newsroom_Bin.csv'
  #file = 'newsroom_autom.csv'
  files = ["train.csv", "val.csv", "test.csv"]
  
  print(datadir)
  #print(file)
  print(files)

  #data_df = pd.read_csv(os.path.join(datadir, file))

  train_df = pd.read_csv(os.path.join(datadir, files[0]))
  val_df = pd.read_csv(os.path.join(datadir, files[1]))
  test_df = pd.read_csv(os.path.join(datadir, files[2]))

  
  return train_df, val_df, test_df


def compute_perplexity_df(data):

    texts = data['SystemSummary'].values
    scores = np.array(compute_perplexity(texts, model='gpt2'))
    data['perplexity'] = scores

    return data


def compute_bleu_df(data):

    summary = data['SystemSummary'].values
    ref_source = data['ArticleText'].values
    ref_title = data['ArticleTitle'].values

    results = []
    for summ, src, title in zip(summary, ref_source, ref_title):
        pred = [summ]
        ref = [[src, title]]
        rsl = compute_bleu(pred, ref)
        results.append(rsl)

    results = np.array(results)

    data['bleu'] = results

    return data

def compute_rouge_df(data):

    summary = data['SystemSummary'].values
    ref_source = data['ArticleText'].values
    ref_title = data['ArticleTitle'].values

    results_rouge1 = []
    results_rouge2 = []
    results_rougeL = []
    for summ, src, title in zip(summary, ref_source, ref_title):
        pred = [summ]
        ref = [[src, title]]
        rouge1, rouge2, rougeL = compute_rouge(pred, ref)
        results_rouge1.append(rouge1)
        results_rouge2.append(rouge2)
        results_rougeL.append(rougeL)

    results_rouge1 = np.array(results_rouge1)
    results_rouge2 = np.array(results_rouge2)
    results_rougeL = np.array(results_rougeL)

    data['rouge1'] = results_rouge1
    data['rouge2'] = results_rouge1
    data['rougeL'] = results_rouge1

    return data

def compute_bleurt_df(data):

    summary = data['SystemSummary'].values
    ref_source = data['ArticleText'].values
    ref_title = data['ArticleTitle'].values

    results_src = compute_bleurt_notune(summary, ref_source)
    results_ttl = compute_bleurt_notune(summary, ref_title)

    #data['bleurt_notune_src'] = np.array(results_src)
    #data['bleurt_notune_ttl'] = np.array(results_ttl)

    data['bleurt20_notune_src'] = np.array(results_src)
    data['bleurt20_notune_ttl'] = np.array(results_ttl)

    return data


def compute_meteor_df(data):

    summary = data['SystemSummary'].values
    ref_source = data['ArticleText'].values
    ref_title = data['ArticleTitle'].values

    results_1 = []
    results_n = []
    for summ, s, t in zip(summary, ref_source, ref_title):
        merg = [s,t]
        results_1.append(compute_meteor([summ], [s]))
        results_n.append(compute_meteor([summ], [merg]))

    data['meteor_1'] = np.array(results_1)
    data['meteor_n'] = np.array(results_n)

    return data


def compute_bertscore_df(data):

    summary = data['SystemSummary'].values
    ref_source = data['ArticleText'].values
    ref_title = data['ArticleTitle'].values

    results_src_precis, results_src_recall, results_src_f1 = compute_bertscore(summary, ref_source)
    results_ttl_precis, results_ttl_recall, results_ttl_f1 = compute_bertscore(summary, ref_title)

    data['bert_f1_src'] = np.array(results_src_precis)
    data['bert_precision_src'] = np.array(results_src_recall)
    data['bert_recall_src'] = np.array(results_src_f1)

    data['bert_f1_ttl'] = np.array(results_ttl_precis)
    data['bert_precision_ttl'] = np.array(results_ttl_recall)
    data['bert_recall_ttl'] = np.array(results_ttl_f1)


    return data


if __name__ == "__main__":
    args = parse_args()

    # read data
    data_df = read_data(args)


    
    # compute perplexity 
    print("Computing Perplexity...")
    new_df = compute_bleu_df(data_df)
    

    # compute BLEU

    print("Computing BLEU...")
    new_df = compute_bleu_df(new_df)
    

    # computing ROUGE
    print("Computing ROUGE...")
    new_df = compute_rouge_df(new_df)

    

    # computing BERTScore 

    
    print("Computing BERTScore...")
    new_df = compute_bertscore_df(new_df)

    #outdir = os.path.join(args.output_dir, 'uber_%s.csv'%(args.attribute))
    outdir = os.path.join(args.output_dir, 'newsroom_auto.csv')
    new_df.to_csv(outdir)
    

   


    
