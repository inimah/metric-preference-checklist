#!/bin/bash
source /<HOME-DIR>/miniconda3/etc/profile.d/conda.sh
conda activate <ENV-NAME>



python auto_metrics/autom_ctrldiag.py \
--output_dir ./data/diag
