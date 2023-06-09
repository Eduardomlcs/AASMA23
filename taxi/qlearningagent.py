from taxi import Agent
import numpy as np
import random

class QLearningAgent(Agent):
    alpha = 0.1  # Learning rate
    gamma = 1.0  # Discount rate
    epsilon = 0.1  # Exploration rate
    
    
    def __init__(self, id, env):
        super(QLearningAgent, self).__init__("QLearning Agent")
        self.id = id
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])

    def action(self,state) -> int:
        if random.uniform(0, 1) < self.epsilon:
            "Basic exploration [~0.47m]"
            return np.random.randint(0,5) # Sample random action (exploration)
            
            "Exploration with action mask [~1.52m]"
          # action = env.action_space.sample(env.action_mask(state)) "Exploration with action mask"
        else:      
            "Exploitation with action mask [~1m52s]"
           # action_mask = np.where(info["action_mask"]==1,0,1) # invert
           # masked_q_values = np.ma.array(q_table[state], mask=action_mask, dtype=np.float32)
           # action = np.ma.argmax(masked_q_values, axis=0)

            "Exploitation with random tie breaker [~1m19s]"
          #  action = np.random.choice(np.flatnonzero(q_table[state] == q_table[state].max()))
            
            "Basic exploitation [~47s]"
            return np.argmax(self.q_table[state]) # Select best known action (exploitation)

    def update(self,action,state, next_state,reward):
        old_q_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        
        new_q_value = (1 - self.alpha) * old_q_value + self.alpha * (reward + self.gamma * next_max)
        
        self.q_table[state, action] = new_q_value

    
 