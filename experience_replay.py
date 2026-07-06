from collections import deque
import random


class ReplayMemory():
    
    #Create Replay Memory with FIFO queue -> Experience Replay
    def __init__(self, maxlen, seed=None):
        self.memory=deque([], maxlen=maxlen)

    def append(self, new_exp):
        self.memory.append(new_exp) #Add a new experience to the memory

    def sample(self, sample_size):
        return random.sample(self.memory,sample_size) #return random samples

    #Current buffer size
    def __len__(self):
        return len(self.memory) #Return the size of the memory