#!/bin/bash

python ./legged_gym/scripts/train.py \
    --task=go1 \
    --num_envs=1024 \
    --headless \
    --sim_device=cpu \
    --rl_device=cpu \
    --ang_vel_xy=-0.1 \
    --torques=-0.0003 \
    --base_height_target=2 \
    --activation='relu' \
    --experiment_name='go1' \
    --run_name='test'
