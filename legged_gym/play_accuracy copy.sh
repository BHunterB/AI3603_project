#!/bin/bash

python ./legged_gym/scripts/play.py \
    --task=go1 \
    --num_envs=1024 \
    --experiment_name=go1 \
    --load_run=Dec29_08-39-11_lijun \
    --episode_length_s=3 \
    --decimation=1 \
    --headless \
    --rl_device=cpu