#!/bin/bash

python ./legged_gym/scripts/play.py \
    --task=go1 \
    --num_envs=50 \
    --experiment_name=go1 \
    --load_run=Nov23_06-52-51_lijun_tlv=1.5_tav=0.75 \
    --episode_length_s=3