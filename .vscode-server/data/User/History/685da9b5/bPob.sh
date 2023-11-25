#!/bin/bash

python ./legged_gym/scripts/train.py \
    --task=go1 \
    --num_envs=1024 \
    --headless \
    --termination=-0.0 \
    --tracking_lin_vel=1.0 \
    --tracking_ang_vel=0.5 \
    --lin_vel_z=-2.0 \
    --ang_vel_xy=-0.05 \
    --orientation=-0. \
    --torques=-0.00001 \
    --dof_vel=-0. \
    --dof_acc=-2.5e-7 \
    --base_height=-0. \
    --feet_air_time=1.0 \
    --collision=-1. \
    --feet_stumble=-0.0 \
    --action_rate=-0.01 \
    --stand_still=-0. \
    --tracking_sigma=0.10 \

    --run_name=lijun \
    --experiment_name=go1
