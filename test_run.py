import argparse

import numpy as np
from gym import Env
import random

from taxi import RandomAgent
from taxi import HeuristicAgent
from taxi.taxi_env import taxi_env

def coordinate_actions(valid_actions, action, taxi1: HeuristicAgent, taxi2: HeuristicAgent):
    new_action = action
    if action not in valid_actions:
        if taxi1.short_distance[0] < taxi2.short_distance[0]:
            new_action = valid_actions[taxi1.short_distance[1]]
        else:
            new_action = valid_actions[taxi2.short_distance[1]]
    return new_action


def run_random_agent(env: taxi_env.TaxiEnv,state, taxi1: RandomAgent, taxi2: RandomAgent):
    
    taxi1.see(state)
    taxi2.see(state)
    action = (taxi1.action(),taxi2.action())
    print("Action: ", action)
    next_state,r,t,_,next_info = env.step(action)
    state = next_state
    info = next_info
    # perform this action on the environment
    env.render()


# create Taxi environment
env = taxi_env.TaxiEnv(render_mode="human")

# create a new instance of taxi, and get the initial state

num_steps = 99
observations = env.reset()
state,info = observations
taxi1 = HeuristicAgent(1)
taxi2 = HeuristicAgent(2)
for s in range(num_steps+1):
    print(f"step: {s} out of {num_steps}")
    taxi1.see(state)
    taxi2.see(state)
    action = (taxi1.action(),taxi2.action())
    print("Action: ", action)
    if action not in info["valid_actions"] or taxi1.step == taxi2.step:
        action = random.choice(info["valid_actions"])
        print("New Action: ", action)
    #print("Before Action: ", action)
    #up_action = coordinate_actions(info["valid_actions"],action,taxi1,taxi2)
    
    next_state,r,t,_,next_info = env.step(action)
    state = next_state
    info = next_info
    # perform this action on the environment
    env.render()
    #run desired agent
   

# end this instance of the taxi environment
env.close()