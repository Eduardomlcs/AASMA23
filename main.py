import gymnasium as gym
import numpy as np
import random
from environment import environment

# create Taxi environment
env = environment.TaxiEnv(render_mode="human")

# create a new instance of taxi, and get the initial state
state = env.reset()
num_steps = 99
for s in range(num_steps+1):
    print(f"step: {s} out of {num_steps}")

    # sample a random action from the list of available actions
    actionA = int(random.randint(0,5))
    actionB = int(random.randint(0,5))

    action = (actionA,actionB)

    print("Action: ", action)

    # perform this action on the environment
    env.step(action)

    env.render()

# end this instance of the taxi environment
env.close()

