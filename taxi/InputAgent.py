from taxi import Agent
import numpy as np

class InputAgent(Agent):

    def __init__(self, id):
        super(InputAgent, self).__init__("Input Agent")
        self.id = id

    def action(self) -> int:
        print("Select Agent"+str(self.id)+" action:")
        return int(input())