import argparse

import numpy as np
import random

from taxi import RandomAgent, InputAgent
from taxi.taxi_env import taxi_env

# create Taxi environment
env = taxi_env.TaxiEnv(render_mode="human")

# create a new instance of taxi, and get the initial state
state,info = env.reset()
num_steps = 99
taxi = InputAgent.InputAgent(1)
taxi2 = InputAgent.InputAgent(2)
for s in range(num_steps+1):
    print(f"step: {s} out of {num_steps}")
    taxi.see(state)
    taxi2.see(state)
    action1 = taxi.action()
    action2 = taxi2.action()
    action = (action1,action2)
    print("Action: ", action)
    next_state,r,t,_,next_info = env.step(action)
    # perform this action on the environment
    env.render()
    state = next_state
   

# end this instance of the taxi environment
env.close()