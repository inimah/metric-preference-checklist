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

task = 'dialogue'

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
            #keyword_dict[nm] = str(nm) + ': ' +  ' '.join(kws)
            
            # for CTRL data
            if nm == 'computers':
                keyword_dict['technologies'] = str(nm) + ': ' + ' '.join(kws)
            else:
                keyword_dict[nm] = str(nm) + ': ' + ' '.join(kws)
            
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
    
    data_df = read_data('./data', 'usr_topical_chat.csv') # in-domain data
    #data_df = read_data('./data', 'usr_persona_chat.csv') 
    
    src_list  = data_df.fact.values
    context_list  = data_df.context.values
    output_list  = data_df.response.values
    

    understandability_sc = []
    naturalness_sc = []
    coherence_sc = []
    engagingness_sc = []
    groundedness_sc = []
    overall_sc = []

    i =0
    for src, ctx, out in zip(src_list, context_list, output_list):
        src_l = [src]
        ctx_l = [ctx]
        out_l = [out]

        # Prepare data for pre-trained evaluators
        data = convert_to_json(output_list=out_l, 
                           src_list=src_l, context_list=ctx_l)
        # Initialize evaluator for a specific task
        evaluator = get_evaluator(task)
        # Get multi-dimensional evaluation scores
        eval_scores = evaluator.evaluate(data, dims=['understandability', 'naturalness', 'coherence', 'engagingness', 'groundedness'],  overall=True, print_result=True)

        if i==0:
            print(eval_scores)
            sys.stdout.flush()

        understandability_sc.append(eval_scores[0]['understandability'])
        naturalness_sc.append(eval_scores[0]['naturalness'])
        coherence_sc.append(eval_scores[0]['coherence'])
        engagingness_sc.append(eval_scores[0]['engagingness'])
        groundedness_sc.append(eval_scores[0]['groundedness'])
        overall_sc.append(eval_scores[0]['overall'])

        i+=1

    data_df['understandability_unieval'] = np.array(understandability_sc)
    data_df['naturalness_unieval'] = np.array(naturalness_sc)
    data_df['coherence_unieval'] = np.array(coherence_sc)
    data_df['engagingness_unieval'] = np.array(engagingness_sc)
    data_df['groundedness_unieval'] = np.array(groundedness_sc)
    data_df['overall_unieval'] = np.array(overall_sc)


    outdir = './data'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    data_df.to_csv(os.path.join(outdir, 'unieval_topical_chat.csv')) # designated data
    
    

    