[![python](https://img.shields.io/badge/Python-3.7.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![jupyter](https://img.shields.io/badge/Jupyter-Lab-F37626.svg?style=flat&logo=Jupyter)](https://jupyterlab.readthedocs.io/en/stable)
[![pytorch](https://img.shields.io/badge/PyTorch-1.11.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

Code implementation and Datasets for the ACL2023 Paper **"NLG Evaluation Metrics Beyond Correlation Analysis: An Empirical Metric Preference Checklist"** [ArxiV](https://arxiv.org/abs/2305.08566)

## Contents

* [Prerequisites](#prerequisites)
* [Quick Start](#quick-start)
  * [1 Structuring Human Evaluation Data](#1-structuring-data)
  * [2 Human-aligned Metrics](#2-human-aligned-metrics)
  * [3 Transfer Experiment](#3-transfer-experiment) 
  * [4 Aspect-level Evaluation](#4-aspect-evaluation)
  * [5 System-level Evaluation](#5-system-evaluation)
  * [6 Pairwise Comparison](#6-pairwise-comparison)
* [Computing Infrastructure](#computing-infrastructure)
* [Citation](#citation)

## Prerequisites

```bash
conda create -n nlgeval_env python=3.7
conda activate nlgeval_env
conda install cudatoolkit=10.1 -c pytorch -n nlgeval_env

pip install -r requirements.txt
```

## Quick Start

#### 1. Structuring Data

Datasets we provided in ~/data have included scores based on human and automatic metrics in this study (including human-aligned metrics).<br>

#### Text Summarization
- **SummEval** (Fabbri et al., 2021)<br>
  **Source**      : Text source before summarized by the systems<br>
  **Decoded**     : Systems'generation outputs<br>
  **Ref-n**       : Ground truth human references (11 references are provided)<br>
  **Model-ID**    : See Appendix of the paper or the original paper for more detail information<br>
  **Coherence**   : Coherence rating by human evaluators (scale 1-5)<br>
  **Consistency** : Consistency rating by human evaluators (scale 1-5)<br>
  **Fluency**     : Fluency rating by human evaluators (scale 1-5)<br>
  **Relevance**   : Relevance rating by human evaluators (scale 1-5)<br>
  **Perplexity**   : Perplexity score for the given output (based on pretrained Language Model)<br>
  **BLEU-n**      : BLEU score for the given output<br>
  **ROUGE-n**     : ROUGE score for the given output<br>
  **BERTScore**   : BERTscore for the given output (Precision, Recall, F1)<br>
  **CTC** 	       : CTC scores (Method: Embedding-based, Discriminative, Regression; Aspect: Consistency, Relevance)<br>
  **CtrlEval**    : CtrlEval scores (Aspect: Coherence)<br>
  **UniEval**     : UniEval scores (Aspect: Coherence, Consistency, Fluency, Relevance, Overall)<br><br>
  

- **Newsroom** (Grusky et al., 2018)<br>
  This dataset is not accompanied with ground truth references. So, for measuring the performance with reference-based metrics or nearly reference-less metrics, we use the source (ArticleText) as a means of reference.<br>
  **ArticleID**   : The unique ID of the article<br>
  **ArticleText**      : Text source before summarized by the systems<br>
  **SystemSummary**     : Systems'generation outputs<br>
  **ArticleTitle**       : Title of the article<br>
  **System**      : NLG System to execute the summarization task. See Appendix of the paper or the original paper for more detail information<br>
  **CoherenceRating**   : Coherence rating by human evaluators (scale 1-5)<br>
  **InformativenessRating** : Informativeness rating by human evaluators (scale 1-5)<br>
  **FluencyRating**     : Fluency rating by human evaluators (scale 1-5)<br>
  **RelevanceRating**   : Relevance rating by human evaluators (scale 1-5)<br>
  **Perplexity**   : Perplexity score for the given output (based on pretrained Language Model)<br>
  **BLEU**      : BLEU score for the given output<br>
  **ROUGE**     : ROUGE score for the given output<br>
  **BERTScore**   : BERTscore for the given output (Precision, Recall, F1)<br>
  **CTC** 	       : CTC scores (Method: Embedding-based, Discriminative, Regression; Aspect: Consistency, Relevance)<br>
  **CtrlEval**    : CtrlEval scores (Aspect: Coherence, Relevance)<br>
  **UniEval**     : UniEval scores (Aspect: Coherence, Consistency, Fluency, Relevance, Overall)<br><br>
  


#### Dialogue Response Generation

- **USR-Topical Chat** (Mehri and Eskenazi, 2020)<br>
  **Fact**   : The factual context of the article<br>
  **Context**      : The preceding conversation as the context for responses<br>
  **Response**     : Responses from the systems or human<br>
  **Annotators**       : The annotator for the corresponding human ratings<br>
  **Model**      : NLG System to execute the response generation task. See Appendix of the paper or the original paper for more detail information<br>
  **Understandable**   : Understandable rating by human evaluators (binary scale 0/1, 0=not understandable, 1=understandable)<br>
  **Natural** : Naturalness rating by human evaluators (scale 1-3, 1=not natural, 2=somewhat/moderate, 3=good)<br>
  **MaintainsContext**     : Rating by human evaluators for maintaining context (scale 1-3)<br>
  **Engaging**   : Engagingness rating by human evaluators (scale 1-3)<br>
  **UsesKnowledge**   : Engagingness rating by human evaluators (binary scale 0/1)<br>
  **Overall**   : Overall rating by human evaluators (scale 1-5)<br>
  **Perplexity**   : Perplexity score for the given output (based on pretrained Language Model)<br>
  **BLEU**      : BLEU score for the given output<br>
  **ROUGE**     : ROUGE score for the given output<br>
  **BERTScore**   : BERTscore for the given output (Precision, Recall, F1)<br>
  **CTC** 	       : CTC scores (Method: Embedding-based, Discriminative, Regression; Aspect: Engagingness, Groundedness)<br>
  **CtrlEval**    : CtrlEval scores (Aspect: Coherence, Consistency, Relevance)<br>
  **UniEval**     : UniEval scores (Aspect: Understandability, Naturalness, Coherence, Engagingness, Groundedness, Overall)<br><br>
  
- **USR Persona Chat** (Mehri and Eskenazi, 2020)<br>
  **Fact**   : Persona context of the article<br>
  **Context**      : The preceding conversation as the context for responses<br>
  **Response**     : Responses from the systems or human<br>
  **Annotators**       : The annotator for the corresponding human ratings<br>
  **Model**      : NLG System to execute the response generation task. See Appendix of the paper or the original paper for more detail information<br>
  **Understandable**   : Understandable rating by human evaluators (binary scale 0/1, 0=not understandable, 1=understandable)<br>
  **Natural** : Naturalness rating by human evaluators (scale 1-3, 1=not natural, 2=neutral/moderate, 3=good)<br>
  **MaintainsContext**     : Rating by human evaluators for maintaining context (scale 1-3)<br>
  **Engaging**   : Engagingness rating by human evaluators (scale 1-3)<br>
  **UsesKnowledge**   : Engagingness rating by human evaluators (binary scale 0/1)<br>
  **Overall**   : Overall rating by human evaluators (scale 1-5)<br>
  **Perplexity**   : Perplexity score for the given output (based on pretrained Language Model)<br>
  **BLEU**      : BLEU score for the given output<br>
  **ROUGE**     : ROUGE score for the given output<br>
  **BERTScore**   : BERTscore for the given output (Precision, Recall, F1)<br>
  **CTC** 	       : CTC scores (Method: Embedding-based, Discriminative, Regression; Aspect: Engagingness, Groundedness)<br>
  **CtrlEval**    : CtrlEval scores (Aspect: Coherence, Consistency, Relevance)<br>
  **UniEval**     : UniEval scores (Aspect: Understandability, Naturalness, Coherence, Engagingness, Groundedness, Overall)<br><br>

#### Controlled Generation
- **UBER-PPLM** ((Dathathri et al., 2020))<br>
  This dataset is an open-ended task (no ground truth references).<br>
  **Prefix**      : A word (two words) at the beginning of the sentence as a cue for Language Model to continue the word(s) and complete them into a sentence or full text <br>
  **Text**     : Systems'generation outputs<br>
  **Domain**     : Topic category as a control attribute<br>
  **Annotator**       : The annotator for the corresponding human ratings<br>
  **Model**    : NLG System as text generator. See Appendix of the paper or the original paper for more detail information<br>
  **Pairtxt**  : Model pair given to the annotators<br>
  **Fluency**     : Fluency rating by human evaluators (scale 1-5)<br>
  **Relevance**   : Relevance rating by human evaluators (binary scale 0/1)<br>
  **Perplexity**   : Perplexity score for the given output (based on pretrained Language Model)<br>
  **BERTScore**   : BERTscore for the given output (Precision, Recall, F1)<br>
  **CTC** 	       : CTC scores (Method: Embedding-based, Discriminative, Regression; Aspect: Consistency, Relevance)<br>
  **CtrlEval**    : CtrlEval scores (Aspect: Coherence, Consistency, Relevance)<br>
  **UniEval**     : UniEval scores (Aspect: Coherence, Consistency, Fluency, Relevance, Overall)<br><br>
  
- **CTRL** (Keskar et al., 2019)<br>
  This dataset is an open-ended task (no ground truth references).<br>
  **Prefix**      : A word (two words) at the beginning of the sentence as a cue for Language Model to continue the word(s) and complete them into a sentence or full text <br>
  **Text**     : Systems'generation outputs<br>
  **Domain**     : Topic category as a control attribute<br>
  **Annotator**       : The annotator for the corresponding human ratings<br>
  **Model**    : NLG System as text generator. See Appendix of the paper or the original paper for more detail information<br>
  **Pairtxt**  : Model pair given to the annotators<br>
  **Fluency**     : Fluency rating by human evaluators (scale 1-5)<br>
  **Relevance**   : Relevance rating by human evaluators (binary scale 0/1)<br>
  **Perplexity**   : Perplexity score for the given output (based on pretrained Language Model)<br>
  **BERTScore**   : BERTscore for the given output (Precision, Recall, F1)<br>
  **CTC** 	       : CTC scores (Method: Embedding-based, Discriminative, Regression; Aspect: Consistency, Relevance)<br>
  **CtrlEval**    : CtrlEval scores (Aspect: Coherence, Consistency, Relevance)<br>
  **UniEval**     : UniEval scores (Aspect: Coherence, Consistency, Fluency, Relevance, Overall)<br><br>
  
- **CTRL-Eval** (Ke et al., 2022) <br>
  This dataset is an open-ended task (no ground truth references).<br>
  **Prefix**      : A word (two words) at the beginning of the sentence as a cue for Language Model to continue the word(s) and complete them into a sentence or full text <br>
  **Text**     : Systems'generation outputs<br>
  **Attribute**     : Topic category as a control attribute<br>
  **Coherence**   : Coherence rating by human evaluators (scale 1-5)<br>
  **Consistency** : Consistency rating by human evaluators (scale 1-5)<br>
  **Relevance**   : Relevance rating by human evaluators (binary scale 0/1)<br>
  **Perplexity**   : Perplexity score for the given output (based on pretrained Language Model)<br>
  **BERTScore**   : BERTscore for the given output (Precision, Recall, F1)<br>
  **CTC** 	       : CTC scores (Method: Embedding-based, Discriminative, Regression; Aspect: Consistency, Relevance)<br>
  **CtrlEval**    : CtrlEval scores (Aspect: Coherence, Consistency, Relevance)<br>
  **UniEval**     : UniEval scores (Aspect: Coherence, Consistency, Fluency, Relevance, Overall)<br><br>

#### 2. Human-aligned Metrics

We consider three (3) metrics under this category. Prior to computing the evaluation scores of the given system outputs (above datasets), the following Python implementation of the metrics need to be installed.<br>
- **CTC** (Deng et al., 2021)<br>
  [https://github.com/tanyuqian/ctc-gen-eval](https://github.com/tanyuqian/ctc-gen-eval)<br><br>
- **CTRLEval** (Ke et al., 2022)<br>
  [https://github.com/thu-coai/ctrleval](https://github.com/thu-coai/ctrleval)<br><br>
- **UniEval** (Zhong et al., 2022)<br>
  [https://github.com/maszhongming/unieval](https://github.com/maszhongming/unieval)<br><br>
  
Datasets we provided in ~/data have included scores based on human and automatic metrics in this study (including human-aligned metrics).<br>
However, if you would like to run the automatic metrics on your own datasets, you can see below examples of code implementation.<br>
Prior to running the following scripts, do not forget to modify the environment name in the script.<br>

| Automatic Metric                                                |  Benchmark               |   Bash script                                                |
| ---------------------------------------------------- | ------------------------ | ------------------------------------------------------------ |
| Perplexity, BLEU, ROUGE, BERTScore                   | Text Summarization |  scripts/run_autom_newsroom.sh                           |
| Perplexity, BLEU, ROUGE, BERTScore                   | Controlled Generation    |  scripts/run_autom_uber.sh                                |
| UniEval                                       |  Text Summarization |  scripts/run_unieval_summ.sh                   |
| UniEval                                       |  Dialogue Generation |  scripts/run_unieval_tc.sh                   |

#### 3. Transfer Experiment

```
\notebooks\Plot Transfer Correlation.ipynb
```

#### 4. Aspect Evaluation

```
\notebooks\Quality-Eval.ipynb
```

#### 5. System Evaluation

```
\notebooks\System-Eval.ipynb
```

#### 6. Pairwise Comparison

```
\notebooks\Pairwise_System_Ranking.ipynb
```

## Computing Infrastructure

- GPU: ASUS Turbo GeForce GTX 1080 Ti ( RAM, 3584 CUDA cores, compute capability 6.1); CPU Intel Xeon Broadwell-EP 2683v4 @ 2.1GHz (64 hyperthreads, RAM: 1024GB).
- OS: Ubuntu 16.04.7 LTS (GNU/Linux 4.4.0-138-generic x86_64)

## Citation
