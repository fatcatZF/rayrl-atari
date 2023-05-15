from simulations.player import Player
from simulations.training.replay_buffer import ReplayBuffer

from models.dqn import DQN 

import torch.optim as optim 

import argparse
import os 
import time 
import datetime

import json 

import ray


parser = argparse.ArgumentParser()

parser.add_argument("--gamma", type=float, default=0.99,
                   help="discount factor.")
parser.add_argument("--env-name", type=str, default="ALE/Breakout-v5",
                    help="Name of Environment.")
parser.add_argument("--num-actions", type=int, default=4,
                    help="Number of actions.")
parser.add_argument("--replay-size", type=int, default=8000,
                   help="maximal replay buffer size.")
parser.add_argument("--batch-size", type=int, default=64,
                   help="batch size of replay.")
parser.add_argument("--lr", type=float, default=0.0005,
                   help="learning rate.")
parser.add_argument("--save-folder", type=str, default="trained_models",
                   help="Where to save the trained model.")
parser.add_argument("--max-episode", type=int, default=30000,
                   help="Maximal trained episodes.")
parser.add_argument("--eps-max", type=float, default=1.0, 
                    help="Max epsilon of epsilon greedy.")
parser.add_argument("--eps-min", type=float, default=0.1,
                    help="Min epsilon of epsilon greedy.")
parser.add_argument("--eps-steps", type=float, default=5000.0,
                    help="Epsilon greedy steps.")
parser.add_argument("--tupdate-freq", type=int, default=80,
                    help="target update frequence.")

args = parser.parse_args()

if args.save_folder:
    exp_count = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
    os.mkdir(save_folder)
    info_file = os.path.join(save_folder, "info.json")
    policy_file = os.path.join(save_folder, "policy.pt")
    target_file = os.path.join(save_folder, "target.pt")
    log_file = os.path.join(save_folder, "log.txt")
    log = open(log_file, 'w')


def train_policy():

    policy = DQN(num_actions=args.num_actions) #initialize a policy
    target = DQN(num_actions=args.num_actions) #initialize a target
    target.load_state_dict(policy.state_dict()) 

    player = Player(env_name=args.env_name, frame_width=84, 
                    frame_height=84, history=4)
    
    num_episodes = args.max_episode
    eps = args.eps_max 
    eps_delta = (args.eps_max-args.eps_min)/args.eps_steps

    




        


