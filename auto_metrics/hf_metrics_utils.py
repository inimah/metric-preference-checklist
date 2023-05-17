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

from evaluate import load

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Device: ",device)


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


def compute_perplexity(predictions, model='gpt2'):

	perplexity = load("perplexity", module_type="metric")
	result = perplexity.compute(predictions=predictions, model_id=model)

	return result["perplexities"]


def compute_bleu(predictions, references):

    bleu = load("bleu")
    result = bleu.compute(predictions=predictions, references=references)['bleu']

    return result


def compute_rouge(predictions, references):

    rouge = load('rouge')
    result = rouge.compute(predictions=predictions, references=references)
    #{'rouge1': 1.0, 'rouge2': 1.0, 'rougeL': 1.0, 'rougeLsum': 1.0}

    return result['rouge1'], result['rouge2'], result['rougeL']

def compute_glue(prediction, reference):

    google_bleu = load("google_bleu")
    result = google_bleu.compute(predictions=[prediction], references=[[reference]])

    return result['google_bleu']


def compute_bertscore(predictions, references):

    bertscore = load("bertscore")
    result = bertscore.compute(predictions=predictions, references=references, lang="en")

    return result['precision'], result['recall'], result['f1']

def compute_meteor(predictions, references):

    meteor = load('meteor')
    result = meteor.compute(predictions=predictions, references=references)

    return result['meteor']

def compute_bleurt_notune(predictions, references):

    '''
    checkpoint (str): BLEURT checkpoint. Will default to BLEURT-tiny if not specified. Other models that can be chosen are: "bleurt-tiny-128", "bleurt-tiny-512", "bleurt-base-128", "bleurt-base-512", "bleurt-large-128", "bleurt-large-512", "BLEURT-20-D3", "BLEURT-20-D6", "BLEURT-20-D12" and "BLEURT-20".

    '''

    #bleurt = load("bleurt", module_type="metric", checkpoint ="bleurt-base-128")
    bleurt = load("bleurt", module_type="metric", checkpoint ="BLEURT-20")
    result = bleurt.compute(predictions=predictions, references=references)

    return result['scores']

