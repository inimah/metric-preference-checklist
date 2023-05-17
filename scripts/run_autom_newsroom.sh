#!/bin/bash
source /<HOME-DIR>/miniconda3/etc/profile.d/conda.sh
conda activate <ENV-NAME>



python auto_metrics/autom_newsroom.py \
--output_dir ./data/newsroom
