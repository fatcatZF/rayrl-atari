import random 
from collections import deque

import torch 


class ReplayBuffer:
    """Replay Buffer stores the last N transitions"""
    def __init__(self, max_size=5000, history=4, batch_size=32):
        """
        args:
          max_size: the maximal number of stored transitions
          history: the number of frames stacked to create a state
          batch_size: the number of transitions returned in a minibatch
        """
        self.max_size = max_size
        self.history = history
        self.batch_size = batch_size
        self.frames = deque([], maxlen=max_size)
        self.actions = deque([], maxlen=max_size)
        self.next_frames = deque([], maxlen=max_size)
        self.rewards = deque([], maxlen=max_size)
        self.is_dones = deque([], maxlen=max_size)
        self.indices = [None]*batch_size

    def add_experience(self, frame, action, next_frame, reward, is_done):
        """
        form of data: (frame, action, next_frame, reward, is_done)
        frame: torch tensor, shape: [n_channel, width, height]
        action: integer
        next_frame: torch tensor, shape: [n_channel, width, height]
        is_done: whether the next frame is a terminate state
        """
        self.frames.append(frame)
        self.actions.append(action)
        self.next_frames.append(next_frame)
        self.rewards.append(reward)
        self.is_dones.append(is_done)

    def add_batch_experiences(self, experiences):
        """
        args:
          experiences: 
             list of tuples, (frame, action, next_frame, 
             reward, is_dones)
        """ 
        frames, actions, next_frames, rewards, is_dones = list(zip(*experiences))
        self.frames.extend(list(frames))
        self.actions.extend(list(actions))
        self.next_frames.extend(list(next_frames))
        self.rewards.extend(list(rewards))
        self.is_dones.extend(list(is_dones))
    
    def current_state_available(self):
        """
        Check whether the current state is available
        """
        if (len(self.is_dones) < self.history) or (True in list(self.is_dones)[-self.history:]):
            return False
        else:
            return True
        
    
    def get_current_state(self):
        """Create current state if there exists at least 4 non-terminal frames"""
        state = list(self.next_frames)[-self.history]
        state = torch.cat(state, dim=0) #shape: [n_channels/timesteps, width, height]
        return state 
    

    def get_valid_indices(self):
        experience_size = len(self.frames)
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.history, experience_size-1)
                if True in list(self.is_dones)[index-self.history:index]:
                    continue
                break 
            self.indices[i] = index 


    def get_minibatch(self):
        """
        Returns a minibatch
        """
        batch = []
        self.get_valid_indices()

        for idx in self.indices:
            state = list(self.frames)[idx - self.history + 1:idx + 1]
            state = torch.cat(state, dim=0)
            # shape:[n_timesteps, width, height]
            next_state = list(self.next_frames)[idx - self.history + 1:idx + 1]
            next_state = torch.cat(next_state, dim=0)
            # shape:[n_timesteps, width, height]
            action = self.actions[idx]
            reward = self.rewards[idx]
            is_done = self.is_dones[idx]

            batch.append((state, action, next_state, reward, is_done))

        return batch