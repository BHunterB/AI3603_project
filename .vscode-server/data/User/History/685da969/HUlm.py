#!/bin/bash
NUM_ENVS=1024
ANG_VEL_XY=-0.1
TORQUES=-0.0003
BASE_HEIGHT_TARGET=2
ACTIVATION="\'relu\'"
EXPERMENT_NAME='go1'

run_name = f"\'NE{NUM_ENVS}_ANGXY{ANG_VEL_XY}_TORQUES{TORQUES}_BHT{BASE_HEIGHT_TARGET}_ACTI{ACTIVATION.replace{'\'',''}}\'"

print(f"python ./legged_gym/scripts/train.py --task=go1 --num_envs={NUM_ENVS} --headless --ang_vel_xy={ANG_VEL_XY} --torques={TORQUES} --base_height_target={BASE_HEIGHT_TARGET} --activation={ACTIVATION} --experiment_name=\'go1\' --run_name={run_name}")

