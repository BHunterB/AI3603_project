#!/bin/bash

python ./legged_gym/scripts/play.py \
    --task=go1 \
    --num_envs=128 \
    --experiment_name=go1 \
    --load_run=Dec30_01-25-09_lijun \
    --test_agility=true \
    --episode_length_s=1 \
    --decimation=1 \
    --checkpoint=24000
    # --headless \