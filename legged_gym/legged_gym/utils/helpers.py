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
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil
from typing import List

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        #TODO sort by date to handle change of month
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run==-1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint==-1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(load_run, model)
    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint
            
        # policy 
        if args.init_noise_std is not None:
            cfg_train.policy.init_noise_std = args.init_noise_std
        if args.actor_hidden_dims is not None:
            cfg_train.policy.actor_hidden_dims = args.actor_hidden_dims
        if args.critic_hidden_dims is not None:
            cfg_train.policy.critic_hidden_dims = args.critic_hidden_dims
        if args.activation is not None:
            cfg_train.policy.activation = args.activation
            
        # algorithm
        if args.value_loss_coef is not None:
            cfg_train.algorithm.value_loss_coef = args.value_loss_coef
        if args.use_clipped_value_loss is not None:
            cfg_train.algorithm.use_clipped_value_loss = args.use_clipped_value_loss
        if args.clip_param is not None:
            cfg_train.algorithm.clip_param = args.clip_param
        if args.entropy_coef is not None:
            cfg_train.algorithm.entropy_coef = args.entropy_coef
        if args.num_learning_epochs is not None:
            cfg_train.algorithm.num_learning_epochs = args.num_learning_epochs
        if args.num_mini_batches is not None:
            cfg_train.algorithm.num_mini_batches = args.num_mini_batches
        if args.learning_rate is not None:
            cfg_train.algorithm.learning_rate = args.learning_rate
        if args.schedule is not None:
            cfg_train.algorithm.schedule = args.schedule
        if args.gamma is not None:
            cfg_train.algorithm.gamma = args.gamma
        if args.lam is not None:
            cfg_train.algorithm.lam = args.lam
        if args.desired_kl is not None:
            cfg_train.algorithm.desired_kl = args.desired_kl
        if args.max_grad_norm is not None:
            cfg_train.algorithm.max_grad_norm = args.max_grad_norm
        
            


    return env_cfg, cfg_train

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "anymal_c_flat", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
        
        {"name": "--termination", "type": float, "help": "Scale of termination"},
        {"name": "--tracking_lin_vel", "type": float, "help": "Scale of tracking_lin_vel"},
        {"name": "--tracking_ang_vel", "type": float, "help": "Scale of tracking_ang_vel"},
        {"name": "--lin_vel_z", "type": float, "help": "Scale of lin_vel_z"},
        {"name": "--ang_vel_xy", "type": float, "help": "Scale of ang_vel_xy"},
        {"name": "--orientation", "type": float, "help": "Scale of orientation"},
        {"name": "--torques", "type": float, "help": "Scale of torques"},
        {"name": "--dof_vel", "type": float, "help": "Scale of dof_vel"},
        {"name": "--dof_acc", "type": float, "help": "Scale of dof_acc"},
        {"name": "--base_height", "type": float, "help": "Scale of base_height"},
        {"name": "--feet_air_time", "type": float, "help": "Scale of feet_air_time"},
        {"name": "--collision", "type": float, "help": "Scale of collision"},
        {"name": "--feet_stumble", "type": float, "help": "Scale of feet_stumble"},
        {"name": "--action_rate", "type": float, "help": "Scale of action_rate"},
        {"name": "--stand_still", "type": float, "help": "Scale of stand_still"},
        
        {"name": "--only_positive_rewards", "type": bool, "help": "Scale of only_positive_rewards"},
        {"name": "--tracking_sigma", "type": float, "help": "Scale of tracking_sigma"},
        {"name": "--soft_dof_pos_limit", "type": float, "help": "Scale of soft_dof_pos_limit"},
        {"name": "--soft_dof_vel_limit", "type": float, "help": "Scale of soft_dof_vel_limit"},
        {"name": "--soft_torque_limit", "type": float, "help": "Scale of soft_torque_limit"},
        {"name": "--base_height_target", "type": float, "help": "Scale of base_height_target"},
        {"name": "--max_contact_force", "type": float, "help": "Scale of max_contact_force"},
        
        {"name": "--init_noise_std", "type": float, "help": "Standard deviation for initialization noise. Overrides config file if provided."},
        {"name": "--actor_hidden_dims", "type": List[int], "help": "Dimensions of hidden layers in the actor network. Overrides config file if provided."},
        {"name": "--critic_hidden_dims", "type": List[int], "help": "Dimensions of hidden layers in the critic network. Overrides config file if provided."},
        {"name": "--activation", "type": str, "help": "Activation function for the networks. Overrides config file if provided."},
        
        {"name": "--value_loss_coef", "type": float, "help": "Coefficient for the value loss. Overrides config file if provided."},
        {"name": "--use_clipped_value_loss", "type": bool, "help": "Whether to use clipped value loss. Overrides config file if provided."},
        {"name": "--clip_param", "type": float, "help": "Clipping parameter for the surrogate loss. Overrides config file if provided."},
        {"name": "--entropy_coef", "type": float, "help": "Coefficient for the entropy bonus. Overrides config file if provided."},
        {"name": "--num_learning_epochs", "type": int, "help": "Number of learning epochs per training iteration. Overrides config file if provided."},
        {"name": "--num_mini_batches", "type": int, "help": "Number of mini-batches per learning epoch. Overrides config file if provided."},
        {"name": "--learning_rate", "type": float, "help": "Learning rate for the optimizer. Overrides config file if provided."},
        {"name": "--schedule", "type": str, "help": "Learning rate schedule type. Overrides config file if provided."},
        {"name": "--gamma", "type": float, "help": "Discount factor for rewards. Overrides config file if provided."},
        {"name": "--lam", "type": float, "help": "Lambda parameter for Generalized Advantage Estimation (GAE). Overrides config file if provided."},
        {"name": "--desired_kl", "type": float, "help": "Desired Kullback-Leibler (KL) divergence between old and new policies. Overrides config file if provided."},
        {"name": "--max_grad_norm", "type": float, "help": "Maximum allowed gradient norm. Overrides config file if provided."},


        {"name": "--episode_length_s", "type": int, "help": "The episode length in seconds designed for playing."}
        {"name": "--test_agility", "type": bool, "help": "To setup the environment for testing agility when playing."}

    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

    
