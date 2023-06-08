from taxi import Agent
import numpy as np


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

    def heuristic(self,state):
        # Calculate the estimated cost from the current state to the goal state (destination)
        # The state represents the agent's location (x, y) in the environment
        
        taxiA_row,taxiA_col,taxiB_row,taxiB_col,passA_loc,destA_idx,passB_loc,destB_idx= decode(state)
        # Extract the agent's location from the state
        agent_location = (taxiA_row,taxiA_col) if self.id == 1 else (taxiB_row,taxiB_col)
        pass_location = passA_loc if self.id == 1 else passB_loc
        dest_idx = destA_idx if self.id == 1 else destB_idx
        if pass_location == 4 or pass_location == 5:
            # Define the goal state as the passenger's destination (drop-off location)
            goal_state = self.locs[dest_idx]
        else:
            # Define the goal state as the passenger's pickup location
            goal_state = self.locs[pass_location]
        
        # Calculate the Manhattan distance between the agent's location and the goal state
        distance = abs(agent_location[0] - goal_state[0]) + abs(agent_location[1] - goal_state[1])
        
        # Return the estimated cost (heuristic value)
        return distance

    def action(self,info) -> int:
        # Get the valid actions from the environment observation
        valid_actions = info["valid_actions"]
        print(valid_actions)
        valid_states = info["valid_states"]
        # Calculate the heuristic value for each valid action
        heuristic_values = [self.heuristic(state) for state in valid_states]
        print(heuristic_values)
        # Choose the action with the lowest heuristic value
        actions = valid_actions[np.argmin(heuristic_values)]
        print(actions)
        action = actions[0] if self.id == 1 else actions[1]
        return action