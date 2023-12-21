#!/bin/bash

python ./legged_gym/scripts/play.py \
    --task=go1 \
    --num_envs=1024 \
    --experiment_name=go1 \
    --load_run=Dec21_15-19-06_lijun_lvz=-1.0 \
    --test_agility=true \
    --episode_length_s=1 \
    --headless