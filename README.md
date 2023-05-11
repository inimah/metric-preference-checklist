[![python](https://img.shields.io/badge/Python-3.7.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![jupyter](https://img.shields.io/badge/Jupyter-Lab-F37626.svg?style=flat&logo=Jupyter)](https://jupyterlab.readthedocs.io/en/stable)
[![pytorch](https://img.shields.io/badge/PyTorch-1.11.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

Code implementation and Datasets for the ACL2023 Paper **"NLG Evaluation Metrics Beyond Correlation Analysis: An Empirical Metric Preference Checklist"**

### Contents

* [Prerequisites](#prerequisites)
* [Computing Infrastructure](#computing-infrastructure)
* [Quick Start](#quick-start)
  * [1 Structuring Human Evaluation Data](#1-structuring-data)
  * [2 Transfer Experiment](#2-transfer-experiment) 
  * [3 Aspect-level Evaluation](#3-aspect-eval)
  * [4 System-level Evaluation](#4-system-eval)
* [Citation](#citation)

## Prerequisites

```bash
conda create -n nlgeval_env python=3.7
conda activate nlgeval_env
conda install cudatoolkit=10.1 -c pytorch -n nlgeval_env

pip install -r requirements.txt
```

## Computing Infrastructure

- GPU: ASUS Turbo GeForce GTX 1080 Ti ( RAM, 3584 CUDA cores, compute capability 6.1); CPU Intel Xeon Broadwell-EP 2683v4 @ 2.1GHz (64 hyperthreads, RAM: 1024GB).
- OS: Ubuntu 16.04.7 LTS (GNU/Linux 4.4.0-138-generic x86_64)

## Quick Start

#### 1. Structuring Data

#### 2. Transfer Experiment

#### 3. Aspect Evaluation

#### 4. System Evaluation

### Citation
