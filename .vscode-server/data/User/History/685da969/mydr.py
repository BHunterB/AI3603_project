#!/bin/bash
NUM_ENVS=1024
ANG_VEL_XY=-0.1
TORQUES=-0.0003
BASE_HEIGHT_TARGET=2
ACTIVATION='relu'
EXPERMENT_NAME='go1'

run_name = f"NE{NUM_ENVS}_ANGXY{ANG_VEL_XY}_TORQUES{TORQUES}_BHT{BASE_HEIGHT_TARGET}_ACTI{ACTIVATION}"

print(f"python ./legged_gym/scripts/train.py \
    --task=go1 \
    --num_envs=1024 \
    --headless \
    --ang_vel_xy=-0.1 \
    --torques=-0.0003 \
    --base_height_target=2 \
    --activation='relu' \
    --experiment_name='go1' \
    --run_name='test'")

