# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
from datetime import datetime
from typing import Tuple
import torch
import numpy as np

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}
    
    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg
    
    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg
    
    def update_scale(self, env_cfg, args):
        if args.termination is not None:
            env_cfg.rewards.scales.termination = args.termination
        if args.tracking_lin_vel is not None:
            env_cfg.rewards.scales.tracking_lin_vel = args.tracking_lin_vel
        if args.tracking_ang_vel is not None:
            env_cfg.rewards.scales.tracking_ang_vel = args.tracking_ang_vel
        if args.lin_vel_z is not None:
            env_cfg.rewards.scales.lin_vel_z = args.lin_vel_z
        if args.ang_vel_xy is not None:
            env_cfg.rewards.scales.ang_vel_xy = args.ang_vel_xy
        if args.orientation is not None:
            env_cfg.rewards.scales.orientation = args.orientation
        if args.torques is not None:
            env_cfg.rewards.scales.torques = args.torques
        if args.dof_vel is not None:
            env_cfg.rewards.scales.dof_vel = args.dof_vel
        if args.dof_acc is not None:
            env_cfg.rewards.scales.dof_acc = args.dof_acc
        if args.base_height is not None:
            env_cfg.rewards.scales.base_height = args.base_height
        if args.feet_air_time is not None:
            env_cfg.rewards.scales.feet_air_time = args.feet_air_time
        if args.collision is not None:
            env_cfg.rewards.scales.collision = args.collision
        if args.feet_stumble is not None:
            env_cfg.rewards.scales.feet_stumble = args.feet_stumble
        if args.action_rate is not None:
            env_cfg.rewards.scales.action_rate = args.action_rate
        if args.stand_still is not None:
            env_cfg.rewards.scales.stand_still = args.stand_still
            
        if args.only_positive_rewards is not None:
            env_cfg.rewards.only_positive_rewards = args.only_positive_rewards
        if args.tracking_sigma is not None:
            env_cfg.rewards.tracking_sigma = args.tracking_sigma
        if args.soft_dof_pos_limit is not None:
            env_cfg.rewards.soft_dof_pos_limit = args.soft_dof_pos_limit
        if args.soft_dof_vel_limit is not None:
            env_cfg.rewards.soft_dof_vel_limit = args.soft_dof_vel_limit
        if args.soft_torque_limit is not None:
            env_cfg.rewards.soft_torque_limit = args.soft_torque_limit
        if args.base_height_target is not None:
            env_cfg.rewards.base_height_target = args.base_height_target
        if args.max_contact_force is not None:
            env_cfg.rewards.max_contact_force = args.max_contact_force
            

    def update_env(self, env_cfg, args):
        if args.episode_length_s is not None:
            env_cfg.env.episode_length_s = args.episode_length_s
        if args.test_agility is not None and args.test_agility == True:
            env_cfg.commands.ranges.lin_vel_x = [5.0,5.0]
    
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """ Creates an environment either from a registered namme or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name' 

        Returns:
            isaacgym.VecTaskPython: The created environment
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # check if there is a registered env with that name
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # load config files
            env_cfg, _ = self.get_cfgs(name)
        # override cfg from args (if specified)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
         
        self.update_scale(env_cfg=env_cfg,args=args)
        self.update_env(env_cfg=env_cfg,args=args)
        # print(env_cfg.rewards.scales.torques)
        # print(env_cfg.rewards.scales.dof_acc)
        # print(env_cfg.rewards.base_height_target)
        
        
        set_seed(env_cfg.seed)
        # parse sim params (convert to dict first)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(   cfg=env_cfg,
                            sim_params=sim_params,
                            physics_engine=args.physics_engine,
                            sim_device=args.sim_device,
                            headless=args.headless)
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default") -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """ Creates the training algorithm  either from a registered namme or from the provided config file.

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example). 
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # if config files are passed use them, otherwise load from the name
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # load config files
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        # override cfg from args (if specified)
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)
        

        if log_root=="default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        
        
        # print(log_file)
        if not train_cfg.runner.resume:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "config.log")
            with open(log_file, "w") as f:
                for key, value in vars(args).items():
                    f.write(f"{key}: {value}\n")
        
        train_cfg_dict = class_to_dict(train_cfg)
        runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)
        #save resume path before creating a new log_dir
        resume = train_cfg.runner.resume
        if resume:
            # load previously trained model
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)
        return runner, train_cfg

# make global task registry
task_registry = TaskRegistry()
