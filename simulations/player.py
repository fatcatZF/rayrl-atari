
from utils import frameProcessor, StateCreator

import torch 
import torch.nn as nn 

import gymnasium as gym

import random



class Player:
    def __init__(self, env_name="ALE/Breakout-v5", frame_width=84,
                 frame_height=84, history=4):
        """
        Simulates rollouts of an environment, given a policy to follow
        args: 
            env_name: atari environment name
            frame_width: width of processed frame
            frame_height: height of processed frame
            history: number of frames used to create state
        """
        self.env = gym.make(env_name)
        self.state_creator = StateCreator(frame_width, frame_height, history)
        self.frame_width = frame_width
        self.frame_height = frame_height

    def rollout(self, policy, explore=True, episilon=0.1, render=False):
        """
        rollout one game episode
        Once the lives of the agent==0, the game ends

        args:
            policy: a string "random" denoting random policy 
                    or a trained neural network returnining Q values
            explore: whether explore
            epsilon: Exploration/Exploitation Trade-Off
            render: Wether render the environment while simulation
        """
        game_reward = 0.
        experiences = []
        _ = self.env.reset()
        if render and self.env.render_mode is not None:
            self.env.render()
        frame, _, terminated, truncated, info = self.env.step(1) # Fire
        lives = info["lives"]
        while not (terminated or truncated): 
            if policy == "random":
               action = self.env.action_space.sample()

            elif isinstance(policy, nn.Module):
                policy.eval()
                frame_processed = frameProcessor(frame, self.frame_height, self.frame_width)
                self.state_creator.add_frame(frame_processed)
                state = self.state_creator.create_state()
                #state shape: [n_channels=1, frame_width, frame_height]
                if (explore and random.uniform(0,1)<episilon):
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        Q = policy(state.unsqueeze(0)) #Compute the Q-Value
                        action = torch.argmax(Q.squeeze()).item()
            else:
                print("Invalid Policy")
                return 
            
            next_frame, reward, terminated, truncated, info = self.env.step(action)
            game_reward += reward
            episode_done = terminated or truncated
            current_lives = info["lives"]

            if current_lives < lives: #blood-1
                frame, _, terminated, truncated, info = self.env.step(1)
                lives = current_lives
                episode_done = True 

            else:
                frame = next_frame 

            
            next_frame_processed = frameProcessor(next_frame, self.frame_height, self.frame_width)

            experiences.append((frame_processed, action, next_frame_processed,
                                reward, episode_done))
            

        return experiences, game_reward
            



