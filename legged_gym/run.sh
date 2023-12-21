#!/bin/bash

python ./legged_gym/scripts/train.py \
    --task=go1 \
    --num_envs=1024 \
    --headless \
    --termination=-0.0 \
    --tracking_lin_vel=12.0 \
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
    --action_rate=-0.1 \
    --stand_still=-2.0 \
    --tracking_sigma=0.25 \
    --run_name=lijun \
    --experiment_name=go1 \
    --learning_rate=7.5e-4 \
    --gamma=0.99 \
    --value_loss_coef=0.5 \
    --lam=0.95 \
    --torques=-0.0002 \
    --max_iterations=1500
    #lr=5.e-4
    #ang_vel_xy=-0.05
    #tracking_sigma=0.25
