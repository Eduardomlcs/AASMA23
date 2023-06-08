from taxi import Agent
import numpy as np

class RandomAgent(Agent):

    def __init__(self, id):
        super(RandomAgent, self).__init__("Random Agent")
        self.id = id

    def action(self) -> int:
        return np.random.randint(0,5)