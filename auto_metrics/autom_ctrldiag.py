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


    parser.add_argument("--domain", type=str, default='rocstories', help="Type of evaluator: rocstories vs. writingprompts.")
    parser.add_argument("--task", type=str, default='binary', help="Classification task: binary vs. multilabel.")
    parser.add_argument('-m', '--metric', nargs='+', default=[], help="Metric as feature.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--nlabels", type=int, default=None, help="Number of multi-labels.")
    args = parser.parse_args()


    return args



def read_data(args):

  datadir = args.output_dir
  #file = 'ctrl_diag.csv'
  #file = 'ctrldiag_autom.csv'
  file = 'ctrlDiag_mod_ctc.csv'
  
  print(datadir)
  print(file)

  data_df = pd.read_csv(os.path.join(datadir, file))
  
  return data_df


def compute_perplexity_df(data):

    texts = data['dialog'].values
    scores = np.array(compute_perplexity(texts, model='gpt2'))
    data['perplexity'] = scores

    return data


def compute_perplex_sent(data):

    texts = data['dialog_sent'].values
    all_scores = []
    for t in texts:
        # t is list of sentences
        print(t)
        sys.stdout.flush()
        scores = np.array(compute_perplexity(t, model='gpt2'))
        mean_sc = np.mean(scores)
        all_scores.append(mean_sc)
    data['mean_perplexity_sent'] = np.array(all_scores)

    return data



def compute_bertscore_df(data):

    texts = data['dialog'].values
    model_persona = data['model_persona'].values
    human_persona = data['human_persona'].values

    results_src_precis, results_src_recall, results_src_f1 = compute_bertscore(texts, model_persona)
    results_ttl_precis, results_ttl_recall, results_ttl_f1 = compute_bertscore(texts, human_persona)

    data['bert_f1_mpers'] = np.array(results_src_precis)
    data['bert_precision_mpers'] = np.array(results_src_recall)
    data['bert_recall_mpers'] = np.array(results_src_f1)

    data['bert_f1_hpers'] = np.array(results_ttl_precis)
    data['bert_precision_hpers'] = np.array(results_ttl_recall)
    data['bert_recall_hpers'] = np.array(results_ttl_f1)


    return data


def comp_relev_bertscore(data):

    model_texts = data['mod_out'].values # text output of model
    human_texts = data['hum_out'].values # text output of human evaluator
    model_persona = data['model_persona'].values
    human_persona = data['human_persona'].values

    mod_prec, mod_rec, mod_f1 = compute_bertscore(model_texts, model_persona)
    hum_prec, hum_rec, hum_f1 = compute_bertscore(human_texts, human_persona)

    data['r_bertscore_precision_mod'] = np.array(mod_prec)
    data['r_bertscore_recall_mod'] = np.array(mod_rec)
    data['r_bertscore_f1_mod'] = np.array(mod_f1)

    data['r_bertscore_precision_hum'] = np.array(hum_prec)
    data['r_bertscore_recall_hum'] = np.array(hum_rec)
    data['r_bertscore_f1_hum'] = np.array(hum_f1)

    return data


def comp_summ_bertscore(data):

    texts = data['dialog_hist'].values # text with \n as character to separating speaker sentences 
    model_persona = data['model_persona'].values
    human_persona = data['human_persona'].values
    all_persona = []
    for mod, hum in zip(model_persona, human_persona):
        merge = mod + "\n" + hum
        all_persona.append(merge)

    prec, rec, f1 = compute_bertscore(texts, all_persona)

    data['s_bertscore_precision'] = np.array(prec)
    data['s_bertscore_recall'] = np.array(rec)
    data['s_bertscore_f1'] = np.array(f1)

    return data


def comp_engage_bertscore(data):

    model_texts = data['mod_out'].values # text output of model
    human_texts = data['hum_out'].values # text output of human evaluator
    

    _, _, mod_f1 = compute_bertscore(model_texts, human_texts) # human output as reference , model output as text to be evaluated

    data['e_bertscore_f1_mod'] = np.array(mod_f1)


    return data

def compute_selfbleu_df(data):

    texts = data['dialog'].values
    tokenized_texts = [t.split() for t in texts]
    weights = {'bigram': (1/2., 1/2.), 'trigram': (1/3., 1/3., 1/3.)}
    self_bleu = SelfBLEU(tokenized_texts, weights)

    results = self_bleu.get_score()
    results_2 = np.array(results['bigram'])
    results_3 = np.array(results['trigram'])

    data['self_bleu2'] = results_2
    data['self_bleu3'] = results_3    

    return data

def compute_selfbleu_sent(data):

    texts = data['dialog_sent'].values
    all_scores2 = []
    all_scores3 = []
    for t in texts:
        tokenized_texts = [s.split() for s in t]
        weights = {'bigram': (1/2., 1/2.), 'trigram': (1/3., 1/3., 1/3.)}
        self_bleu = SelfBLEU(tokenized_texts, weights)
        results = self_bleu.get_score()
        results_2 = np.mean(np.array(results['bigram']))
        results_3 = np.mean(np.array(results['trigram']))
        all_scores2.append(results_2)
        all_scores3.append(results_3)

    data['mean_self_bleu2'] = np.array(all_scores2)
    data['mean_self_bleu3'] = np.array(all_scores3)

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

    print("Computing BERTScore...")
    new_df = compute_bertscore_df(data_df)

    outdir = os.path.join(args.output_dir, 'ctrldiag_autom.csv')
    new_df.to_csv(outdir)

    