from taxi import Agent
from taxi.a_star_heuristic import a_star_search
import numpy as np

MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]

def decode(i):
    out = []
    out.append(i % 4)
    i = i // 4
    out.append(i % 6)
    i = i // 6
    out.append(i % 4)
    i = i // 4
    out.append(i % 6)
    i = i // 6
    out.append(i % 5)
    i = i // 5
    out.append(i % 5)
    i = i // 5
    out.append(i % 5)
    i = i // 5
    out.append(i)
    assert 0 <= i < 6
    return reversed(out)


class HeuristicAgent(Agent):

    def __init__(self,id):
        super(HeuristicAgent, self).__init__("Heuristic Agent")
        self.locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
        self.id = id
        self.map = np.asarray(MAP, dtype="c")

    def heuristic(self):
        # Calculate the estimated cost from the current state to the goal state (destination)
        # The state represents the agent's location (x, y) in the environment
        actions = []
        taxiA_row,taxiA_col,taxiB_row,taxiB_col,passA_loc,destA_idx,passB_loc,destB_idx= decode(self.observation)
        # Extract the agent's location from the state
        self.agent_location = (taxiA_row,taxiA_col) if self.id == 1 else (taxiB_row,taxiB_col)
        pass_location = passA_loc if self.id == 1 else passB_loc
        dest_idx = destA_idx if self.id == 1 else destB_idx
        if  (pass_location != 4 and pass_location != 5) and self.agent_location == self.locs[pass_location]:
            return 0
        if self.agent_location == self.locs[dest_idx] and (pass_location == 4 or pass_location == 5):
            return 1
        if pass_location == 4 or pass_location == 5:
            # Define the goal state as the passenger's destination (drop-off location)
            goal_state = self.locs[dest_idx]
        else:
            # Define the goal state as the passenger's pickup location
            goal_state = self.locs[pass_location]
        
        convert_agent_location = (self.agent_location[0] + 1, self.agent_location[1]*2 + 1)
        convert_goal_state = (goal_state[0] + 1, goal_state[1]*2 + 1)

        path = a_star_search(convert_agent_location,convert_goal_state,self.map)
        
        print("Caminho ",self.id,": ",path)

        """
        for step in path:
            if step[0] - 1 == agent_location[0]:
                actions = actions + [2,]
            elif step[0] + 1 == agent_location[0]:
                actions = actions + [3,]
            elif step[1] - 1 == agent_location[1]:
                actions = actions + [0,]
            elif step[1] + 1 == agent_location[1]:
                actions = actions + [1,]
        """

        # Return the estimated cost (heuristic value)
        return path

    def action(self) -> int:
        path = self.heuristic()
        if path == 0:
            return 4
        if path == 1:
            return 5
        step = path[1]
        self.step = step
        start = path[0]
        if step[0] - 1 == start[0]:
            return 0 #BAIXO
        elif step[0] + 1 == start[0]:
            return 1 #CIMA
        elif step[1] - 1 == start[1]:
            return 2 #DIREITA
        elif step[1] + 1 == start[1]:
            return 3 #ESQUERDA
        