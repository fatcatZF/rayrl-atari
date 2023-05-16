from simulations.player import Player
from simulations.training.replay_buffer import ReplayBuffer

from models.dqn import DQN 

import torch 
import torch.nn as nn
import torch.optim as optim 

import numpy as np 

import argparse
import os 
import time 
import datetime

from collections import deque

import json 

import ray


parser = argparse.ArgumentParser()

parser.add_argument("--gamma", type=float, default=0.99,
                   help="discount factor.")
parser.add_argument("--env-name", type=str, default="ALE/Breakout-v5",
                    help="Name of Environment.")
parser.add_argument("--num-actions", type=int, default=4,
                    help="Number of actions.")
parser.add_argument("--num-sims",type=int, default=4,
                    help="Number of simulations.")
parser.add_argument("--replay-size", type=int, default=8000,
                   help="maximal replay buffer size.")
parser.add_argument("--batch-size", type=int, default=64,
                   help="batch size of replay.")
parser.add_argument("--lr", type=float, default=0.0005,
                   help="learning rate.")
parser.add_argument("--save-folder", type=str, default="simulations/trained_models",
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



PlayActor = ray.remote(Player) # Ray Actor for distributed simulation

def train_policy():

    ray.init() #initialize a cluster

    policy = DQN(num_actions=args.num_actions) #initialize a policy
    target = DQN(num_actions=args.num_actions) #initialize a target
    target.load_state_dict(policy.state_dict()) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    target.to(device)

    # Optimizer for the policy 
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    #Define Loss Criterion
    loss_criterion = nn.SmoothL1Loss()

    # put policy into the object store
    policy_ref = ray.put(policy)
    
    num_episodes = args.max_episode # Number of episodes

    # Episilon-Greedy Configuration 
    eps = args.eps_max 
    eps_delta = (args.eps_max-args.eps_min)/args.eps_steps
    eps_min = args.eps_min

    # Gamma
    gamma = args.gamma 

    # Initialize Replay Buffer 
    replay_buffer = ReplayBuffer(max_size=args.replay_size, history=4, 
                                 batch_size=args.batch_size)
    

    # create players for distributed simulation
    num_sims = args.num_sims 
    players = [PlayActor.remote(env_name=args.env_name, 
                                frame_width=84, 
                                frame_height=84,
                                history=4) for _ in range(num_sims)]
    
    # Player for Evaluation
    player_eval = Player(env_name=args.env_name, 
                         frame_width=84,
                         frame_height=84,
                         history=4)
    
    rewards_last_100_episodes = deque([], maxlen=100)
    
    for episode in range(num_episodes):

        # Simulation And Training Phase
        experiences = [player.rollout.remote(policy_ref, explore=True, episilon=eps) for player in players]
        while len(experiences) > 0:
            finished, experiences = ray.wait(experiences)
            replay_buffer.add_batch_experiences(ray.get(finished)[0][0]) # add finished experiences to the replay buffer
            minibatch = replay_buffer.get_minibatch() # Sample a minibatch from replay buffer
            # list of tuples of form (state, action ,next_state, reward, is_done)
            batch_trans = list(map(list, zip(*minibatch)))
            states = torch.stack(batch_trans[0], dim=0).to(device)
            actions = batch_trans[1]
            batch_size = len(actions)
            batch_indices = range(batch_size)
            next_states = torch.stack(batch_trans[2], dim=0).to(device)
            rewards = torch.tensor(batch_trans[3]).to(device)
            is_dones = torch.tensor(batch_trans[4]).to(device)

            # Predicted Q Values
            policy.train()
            Q_predicted = policy(states)
            # Get predicted Q(s,a)
            Q_predicted = Q_predicted[batch_indices, actions]

            # Get Target Q Values
            with torch.no_grad():
                target.eval()
                Q_next = target(next_states)
                Q_next_max = torch.max(Q_next, -1).values
                Q_next_max[is_dones] = 0.
                Q_target = gamma*Q_next_max+rewards

            # Train Policy
            loss = loss_criterion(Q_predicted, Q_target)
            optimizer.zero_grad()
            loss.backward()
            for param in policy.parameters():
                param.grad.data.clamp_(-1,1)
            optimizer.step()


        # Evaluation Phase
        _, game_reward = player_eval.rollout(policy, explore=False)
        rewards_last_100_episodes.append(game_reward)
        average_reward = np.average(rewards_last_100_episodes)

        print(f"Evaluation of Episode {episode+1}, game_reward: {game_reward}, average reward: {average_reward}")



        # Change the value of eps for exploration
        eps = max(eps-eps_delta, eps_min)


        # Update the target
        if (episode+1)%100 == 0:
            target.load_state_dict(policy.state_dict())





train_policy()




        
        
        
        






    

    






    




        


