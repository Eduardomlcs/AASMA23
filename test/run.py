import argparse

import numpy as np
from gym import Env
import random

from taxi import Agent
from taxi.taxi_env import taxi_env

class RandomAgent(Agent):

    def __init__(self):
        super(RandomAgent, self).__init__("Random Agent")

    def action(self) -> int:
        return np.random.randint(0,5)
# create Taxi environment
env = taxi_env.TaxiEnv(render_mode="human")

# create a new instance of taxi, and get the initial state
state = env.reset()
num_steps = 99
taxi = RandomAgent()
taxi2 = RandomAgent()
for s in range(num_steps+1):
    print(f"step: {s} out of {num_steps}")
    taxi.see(state)
    taxi2.see(state)
    action1 = taxi.action()
    action2 = taxi2.action()
    action = (action1,action2)
    print("Action: ", action)
    test = env.step(action)
     # perform this action on the environment
    env.render()
    state = test[0]
   

# end this instance of the taxi environment
env.close()