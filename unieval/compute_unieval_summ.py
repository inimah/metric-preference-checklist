import os
import sys
sys.path.append(os.getcwd())
import torch 
import numpy as np 
import pandas as pd
import string
import re
import nltk
import argparse



from utils import convert_to_json
from metric.evaluator import get_evaluator

task = 'summarization'

def cleaning_text(text):
    
    punct = ''.join([p for p in string.punctuation])
    
    text = text.replace('<SNT>', '')
    text = re.sub(r'\\', ' ', text)
    text = re.sub(r'\n ', '\n', text)
    text = re.sub(r' +', ' ', text)

    text = text.replace('i.e.', 'id est')
    text = text.replace('e.g.', 'exempli gratia')
    text = text.lower().replace('q&a', 'question and answer')
    text = text.replace('&', 'and')
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[-+]?[.\d]*[\d]+[:,.\d]*', '', text)
    text = re.sub(r'[^\x00-\x7f]', '', text)
    text = re.sub(r'\b\w{1}\b', '', text)
    text = re.sub(r'\b\w{20.1000}\b', '', text)
    regex = re.compile('[%s]' % re.escape(punct)) 
    text = regex.sub(' ', text)

    return text

def read_data(datadir, file):

    data = pd.read_csv(os.path.join(datadir, file))

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
        print(nm)
        sys.stdout.flush()
        if args.attribute == 'topic':
            keyword_dict[nm] = str(nm) + ': ' + ' '.join(kws)
            '''
            # for CTRL data
            if nm == 'computers':
                keyword_dict['technologies'] = str(nm) + ': ' + ' '.join(kws)
            else:
                keyword_dict[nm] = str(nm) + ': ' + ' '.join(kws)
            '''
        else:
            keyword_dict[nm] = ' '.join(kws)

    print(keyword_dict.keys())

    return keyword_dict



def parse_args():
    parser = argparse.ArgumentParser(description="Training a multi-modal multi-label classification")


    parser.add_argument("--attribute", type=str, help="senti vs. topic as control attribute.")
    parser.add_argument("--task", type=str, default='binary', help="Classification task: binary vs. multilabel.")
    parser.add_argument('-m', '--metric', nargs='+', default=[], help="Metric as feature.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--nlabels", type=int, default=None, help="Number of multi-labels.")
    args = parser.parse_args()


    return args



if __name__ == "__main__":

    args = parse_args()

    # read data
    
    data_df = read_data('./data', 'summeval.csv')
    

    src_list  = data_df.source.values
    ref_list   = data_df.ref1.values
    output_list  = data_df.decoded.values
 

    coherence_sc = []
    consistency_sc = []
    fluency_sc = []
    relevance_sc = []
    overall_sc = []

    i =0
    for src, ref, out in zip(src_list, ref_list, output_list):
        src_l = [src]
        ref_l = [ref]
        out_l = [out]

        # Prepare data for pre-trained evaluators
        data = convert_to_json(output_list=out_l, 
                           src_list=src_l, ref_list=ref_l)
        # Initialize evaluator for a specific task
        evaluator = get_evaluator(task)
        # Get multi-dimensional evaluation scores
        eval_scores = evaluator.evaluate(data, dims=['coherence', 'consistency', 'fluency', 'relevance'],  overall=True, print_result=True)

        if i==0:
            print(eval_scores)
            sys.stdout.flush()

        coherence_sc.append(eval_scores[0]['coherence'])
        consistency_sc.append(eval_scores[0]['consistency'])
        fluency_sc.append(eval_scores[0]['fluency'])
        relevance_sc.append(eval_scores[0]['relevance'])
        overall_sc.append(eval_scores[0]['overall'])

        i+=1

    data_df['coherence_unieval'] = np.array(coherence_sc)
    data_df['consistency_unieval'] = np.array(consistency_sc)
    data_df['fluency_unieval'] = np.array(fluency_sc)
    data_df['relevance_unieval'] = np.array(relevance_sc)
    data_df['overall_unieval'] = np.array(overall_sc)


    outdir = './data'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    data_df.to_csv(os.path.join(outdir, 'unieval_summeval.csv'))
    

    