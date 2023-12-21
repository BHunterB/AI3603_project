#!/bin/bash

python ./legged_gym/scripts/play.py \
    --task=go1 \
    --num_envs=2048 \
    --experiment_name=go1 \
    --load_run=DeDec11_17-08-19_lijun_lv=5_av=0.25 \
    --episode_length_s=3 \
    --headless