from contextlib import closing
from io import StringIO
from os import path
from typing import Optional

import numpy as np

import gymnasium as gym
from gymnasium import Env, spaces, utils
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.error import DependencyNotInstalled


MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]
WINDOW_SIZE = (550, 350)


class TaxiEnv(Env):
    """
    The Taxi Problem involves navigating to passengers in a grid world, picking them up and dropping them
    off at one of four locations.
    ## Description
    There are four designated pick-up and drop-off locations (Red, Green, Yellow and Blue) in the
    5x5 grid world. The taxi starts off at a random square and the passenger at one of the
    designated locations.
    The goal is move the taxi to the passenger's location, pick up the passenger,
    move to the passenger's desired destination, and
    drop off the passenger. Once the passenger is dropped off, the episode ends.
    The player receives positive rewards for successfully dropping-off the passenger at the correct
    location. Negative rewards for incorrect attempts to pick-up/drop-off passenger and
    for each step where another reward is not received.
    Map:
            +---------+
            |R: | : :G|
            | : | : : |
            | : : : : |
            | | : | : |
            |Y| : |B: |
            +---------+
    From "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich [<a href="#taxi_ref">1</a>].
    ## Action Space
    The action shape is `(1,)` in the range `{0, 5}` indicating
    which direction to move the taxi or to pickup/drop off passengers.
    - 0: Move south (down)
    - 1: Move north (up)
    - 2: Move east (right)
    - 3: Move west (left)
    - 4: Pickup passenger
    - 5: Drop off passenger
    ## Observation Space
    There are 500 discrete states since there are 25 taxi positions, 5 possible
    locations of the passenger (including the case when the passenger is in the
    taxi), and 4 destination locations.
    Destination on the map are represented with the first letter of the color.
    Passenger locations:
    - 0: Red
    - 1: Green
    - 2: Yellow
    - 3: Blue
    - 4: In taxi
    Destinations:
    - 0: Red
    - 1: Green
    - 2: Yellow
    - 3: Blue
    An observation is returned as an `int()` that encodes the corresponding state, calculated by
    `((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination`
    Note that there are 400 states that can actually be reached during an
    episode. The missing states correspond to situations in which the passenger
    is at the same location as their destination, as this typically signals the
    end of an episode. Four additional states can be observed right after a
    successful episodes, when both the passenger and the taxi are at the destination.
    This gives a total of 404 reachable discrete states.
    ## Starting State
    The episode starts with the player in a random state.
    ## Rewards
    - -1 per step unless other reward is triggered.
    - +20 delivering passenger.
    - -10  executing "pickup" and "drop-off" actions illegally.
    An action that results a noop, like moving into a wall, will incur the time step
    penalty. Noops can be avoided by sampling the `action_mask` returned in `info`.
    ## Episode End
    The episode ends if the following happens:
    - Termination:
            1. The taxi drops off the passenger.
    - Truncation (when using the time_limit wrapper):
            1. The length of the episode is 200.
    ## Information
    `step()` and `reset()` return a dict with the following keys:
    - p - transition proability for the state.
    - action_mask - if actions will cause a transition to a new state.
    As taxi is not stochastic, the transition probability is always 1.0. Implementing
    a transitional probability in line with the Dietterich paper ('The fickle taxi task')
    is a TODO.
    For some cases, taking an action will have no effect on the state of the episode.
    In v0.25.0, ``info["action_mask"]`` contains a np.ndarray for each of the actions specifying
    if the action will change the state.
    To sample a modifying action, use ``action = env.action_space.sample(info["action_mask"])``
    Or with a Q-value based algorithm ``action = np.argmax(q_values[obs, np.where(info["action_mask"] == 1)[0]])``.
    ## Arguments
    ```python
    import gymnasium as gym
    gym.make('Taxi-v3')
    ```
    ## References
    <a id="taxi_ref"></a>[1] T. G. Dietterich, “Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition,”
    Journal of Artificial Intelligence Research, vol. 13, pp. 227–303, Nov. 2000, doi: 10.1613/jair.639.
    ## Version History
    * v3: Map Correction + Cleaner Domain Description, v0.25.0 action masking added to the reset and step information
    * v2: Disallow Taxi start location = goal location, Update Taxi observations in the rollout, Update Taxi reward threshold.
    * v1: Remove (3,2) from locs, add passidx<4 check
    * v0: Initial version release
    """

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.desc = np.asarray(MAP, dtype="c")

        self.locs = locs = [(0, 0), (0, 4), (4, 0), (4, 3)]
        self.locs_colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255)]
        num_states =360000 #25*25*6*6*4*4
        num_rows = 5
        num_columns = 5
        max_row = num_rows - 1
        max_col = num_columns - 1
        self.initial_state_distrib = np.zeros(num_states)
        
        num_actions = 6
        self.actions = actions = []
        for a in range(num_actions):
            for b in range(num_actions):
                self.actions.append(tuple([a,b]))
                
        self.P = {
            state: {action: [] for action in self.actions}
            for state in range(num_states)
        }
        
        for rowA in range(num_rows):
            for colA in range(num_columns):
                for rowB in range(num_rows):
                    for colB in range(num_columns):                
                        for passA_idx in range(len(locs) + 2):  # +1 for being inside taxi
                            for destA_idx in range(len(locs)):
                                for passB_idx in range(len(locs) + 2):  # +1 for being inside taxi
                                    for destB_idx in range(len(locs)):
                                        state = self.encode(rowA, colA,rowB,colB, passA_idx, destA_idx, passB_idx, destB_idx)
                                        if passA_idx < 4 and passB_idx<4 and passA_idx != destA_idx and passB_idx!= destB_idx and (rowA != rowB or colA != colB ):
                                            self.initial_state_distrib[state] += 1
                                        for action in self.actions:
                                            # defaults
                                            new_rowA, new_colA, new_passA_idx = rowA, colA, passA_idx
                                            new_rowB, new_colB, new_passB_idx = rowB, colB, passB_idx
                                            reward = [-1,-1]  # default reward when there is no pickup/dropoff
                                            terminated = False
                                            taxiA_loc = (rowA, colA)
                                            taxiB_loc = (rowB, colB)
                                            #TODO Rewards
                                            if action[0] == 0:
                                                new_rowA = min(rowA + 1, max_row)
                                            elif action[0] == 1:
                                                new_rowA = max(rowA - 1, 0)
                                            elif action[0] == 2 and self.desc[1 + rowA, 2 * colA + 2] == b":":
                                                new_colA = min(colA + 1, max_col)
                                            elif action[0] == 3 and self.desc[1 + rowA, 2 * colA] == b":":
                                                new_colA = max(colA - 1, 0)
                                            elif action[0] == 4:  # pickup
                                                if passA_idx < 4 and taxiA_loc == locs[passA_idx] and passB_idx!=4:
                                                    new_passA_idx = 4
                                                elif passB_idx < 4 and taxiA_loc == locs[passB_idx] and passA_idx!=4:
                                                    new_passB_idx = 4
                                                else:  # passenger not at location
                                                    reward[0] = -10
                                            elif action[0] == 5:  # dropoff PRECISA DE SER ALTERADO
                                                #TODO terminacao do programa
                                                if (taxiA_loc == locs[destA_idx]) and passA_idx == 4:
                                                    new_passA_idx = destA_idx
                                                    terminated = True
                                                    reward[0] = 20
                                                else:  # dropoff at wrong location
                                                    reward[0] = -10
                                                
                                                
                                            if action[1] == 0:
                                                new_rowB = min(rowB + 1, max_row)
                                            elif action[1] == 1:
                                                new_rowB = max(rowB - 1, 0)
                                            elif action[1] == 2 and self.desc[1 + rowB, 2 * colB + 2] == b":":
                                                new_colB = min(colB + 1, max_col)
                                            elif action[1] == 3 and self.desc[1 + rowB, 2 * colB] == b":":
                                                new_colB = max(colB - 1, 0)
                                            elif action[1] == 4:  # pickup
                                                if passA_idx < 4 and taxiB_loc == locs[passA_idx] and passB_idx!=5:
                                                    new_passA_idx = 5
                                                elif passB_idx < 4 and taxiB_loc == locs[passB_idx] and passA_idx!=5:
                                                    new_passB_idx = 5
                                                else:  # passenger not at location
                                                    reward[1] = -10
                                            elif action[1] == 5:  # dropoff PRECISA DE SER ALTERADO
                                                #TODO terminacao do programa
                                                if (taxiB_loc == locs[destB_idx]) and passB_idx == 5:
                                                    new_passB_idx = destB_idx
                                                    terminated = True
                                                    reward[1] = 20
                                                else:  # dropoff at wrong location
                                                    reward[1] = -10
                                                    
                                            #Choques entre carros
                                            if(new_rowA==new_rowB and new_colA==new_colB):
                                                reward[0]= -30
                                                reward[1]= -30 #very negative
                                                new_rowA, new_rowB, new_colA, new_colB = rowA,rowB,colA,colB       
                                            new_state = self.encode(
                                                new_rowA, new_colA,new_rowB,new_colB, new_passA_idx, destA_idx, new_passB_idx, destB_idx
                                            )
                                            self.P[state][action].append(
                                                (1.0, new_state, reward, terminated)
                                            )
        self.initial_state_distrib /= self.initial_state_distrib.sum()
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Discrete(num_states)

        self.render_mode = render_mode

        # pygame utils
        self.window = None
        self.clock = None
        self.cell_size = (
            WINDOW_SIZE[0] / self.desc.shape[1],
            WINDOW_SIZE[1] / self.desc.shape[0],
        )
        self.taxi_imgs = None
        self.taxiB_imgs = None
        self.taxi_orientation = 0
        self.taxiB_orientation = 0
        self.passenger_img = None
        self.passengerB_img = None
        self.destination_img = None
        self.destinationB_img = None
        self.median_horiz = None
        self.median_vert = None
        self.background_img = None

    
    def encode(self,rowA,colA,rowB,colB, passA_idx, destA_idx, passB_idx, destB_idx):# taxi_row, taxi_col, pass_loc, dest_idx
    # (5) 5, 5, 4
        i = rowA
        i*=5
        i+= colA
        i*=5
        i+= rowB
        i*=5
        i+= colB
        i*=6
        i+= passA_idx
        i*=4
        i+= destA_idx
        i*=6
        i+= passB_idx
        i*=4
        i+= destB_idx
        return i

    def decode(self, i):
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


    def action_mask(self, state: int):
        """Computes an action mask for the action space using the state information."""
        mask = np.zeros((2,6), dtype=np.int8)
        taxiA_row,taxiA_col,taxiB_row,taxiB_col, passA_loc, destA_idx,passB_loc,destB_idx= self.decode(state)
        if taxiA_row < 4:
            mask[0][0] = 1
        if taxiA_row > 0:
            mask[0][1] = 1
        if taxiA_col < 4 and self.desc[taxiA_row + 1, 2 * taxiA_col + 2] == b":":
            mask[0][2] = 1
        if taxiA_col > 0 and self.desc[taxiA_row + 1, 2 * taxiA_col] == b":":
            mask[0][3] = 1
        if passA_loc < 4 and (taxiA_row, taxiA_col) == self.locs[passA_loc]:
            mask[0][4] = 1
        if passA_loc == 4 and (
            (taxiA_row, taxiA_col) == self.locs[destA_idx]
            or (taxiA_row, taxiA_col) in self.locs
        ):
            mask[0][5] = 1

        if taxiB_row < 4:
            mask[1][0] = 1
        if taxiB_row > 0:
            mask[1][1] = 1
        if taxiB_col < 4 and self.desc[taxiB_row + 1, 2 * taxiB_col + 2] == b":":
            mask[1][2] = 1
        if taxiB_col > 0 and self.desc[taxiB_row + 1, 2 * taxiB_col] == b":":
            mask[1][3] = 1
        if passB_loc < 4 and (taxiB_row, taxiB_col) == self.locs[passB_loc]:
            mask[1][4] = 1
        if passB_loc == 4 and (
            (taxiB_row, taxiB_col) == self.locs[destB_idx]
            or (taxiB_row, taxiB_col) in self.locs
        ):
            mask[1][5] = 1

        return mask
    
    def valid_actions(self,mask):
        valid = []
        for i in range(2):
            for j in range(6):
                if mask[i][j] == 1:
                    valid = valid + [(i,j),]
        return valid
    
    def valid_states(self,state,valid_actions):
        valid = []
        for action in valid_actions:
            transitions = self.P[state][action]
            i = categorical_sample([t[0] for t in transitions], self.np_random)
            _, s, _, _ = transitions[i]
            valid = valid + [s,]
        return valid


    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        
        self.s = s
        self.lastaction = a
        actions = self.valid_actions(self.action_mask(s))
        states = self.valid_states(s,actions)
        assert len(states) == len(actions)
        #if self.render_mode == "human":
            #self.render()
        return (int(s), r, t, False, {"prob": p, "action_mask": self.action_mask(s), "valid_states": states, "valid_actions": actions})

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None
        self.a_mask = None
        self.taxi_orientation = 0
        self.taxiB_orientation = 0
        actions = self.valid_actions(self.action_mask(self.s))
        states = self.valid_states(self.s,actions)

        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1.0, "action_mask": self.action_mask(self.s), "valid_states": states, "valid_actions": actions}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        elif self.render_mode == "ansi":
            return self._render_text()
        else:  # self.render_mode in {"human", "rgb_array"}:
            return self._render_gui(self.render_mode)

    def _render_gui(self, mode):
        try:
            import pygame  # dependency to pygame only if rendering with human
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy-text]`"
            ) from e

        if self.window is None:
            pygame.init()
            pygame.display.set_caption("Taxi")
            if mode == "human":
                self.window = pygame.display.set_mode(WINDOW_SIZE)
            elif mode == "rgb_array":
                self.window = pygame.Surface(WINDOW_SIZE)

        assert (
            self.window is not None
        ), "Something went wrong with pygame. This should never happen."
        if self.clock is None:
            self.clock = pygame.time.Clock()
            
        if self.taxi_imgs is None:
            file_names = [
                path.join(path.dirname(__file__), "img/cab_front.png"),
                path.join(path.dirname(__file__), "img/cab_rear.png"),
                path.join(path.dirname(__file__), "img/cab_right.png"),
                path.join(path.dirname(__file__), "img/cab_left.png"),
            ]
            self.taxi_imgs = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]
            
        if self.taxiB_imgs is None:
            file_names = [
                path.join(path.dirname(__file__), "img/cab_frontB.png"),
                path.join(path.dirname(__file__), "img/cab_rearB.png"),
                path.join(path.dirname(__file__), "img/cab_rightB.png"),
                path.join(path.dirname(__file__), "img/cab_leftB.png"),
            ]
            
            self.taxiB_imgs = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]
            
        if self.passenger_img is None:
            file_name = path.join(path.dirname(__file__), "img/passenger.png")
            self.passenger_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
            
        if self.passengerB_img is None:
            file_name = path.join(path.dirname(__file__), "img/passengerB.png")
            self.passengerB_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
            
        if self.destination_img is None:
            file_name = path.join(path.dirname(__file__), "img/hotel.png")
            self.destination_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
            self.destination_img.set_alpha(170)
            
        if self.destinationB_img is None:
            file_name = path.join(path.dirname(__file__), "img/hotelB.png")
            self.destinationB_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
            self.destinationB_img.set_alpha(170)
            
        if self.median_horiz is None:
            file_names = [
                path.join(path.dirname(__file__), "img/gridworld_median_left.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_horiz.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_right.png"),
            ]
            self.median_horiz = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]
        if self.median_vert is None:
            file_names = [
                path.join(path.dirname(__file__), "img/gridworld_median_top.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_vert.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_bottom.png"),
            ]
            self.median_vert = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]
        if self.background_img is None:
            file_name = path.join(path.dirname(__file__), "img/taxi_background.png")
            self.background_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        desc = self.desc

        for y in range(0, desc.shape[0]):
            for x in range(0, desc.shape[1]):
                cell = (x * self.cell_size[0], y * self.cell_size[1])
                self.window.blit(self.background_img, cell)
                if desc[y][x] == b"|" and (y == 0 or desc[y - 1][x] != b"|"):
                    self.window.blit(self.median_vert[0], cell)
                elif desc[y][x] == b"|" and (
                    y == desc.shape[0] - 1 or desc[y + 1][x] != b"|"
                ):
                    self.window.blit(self.median_vert[2], cell)
                elif desc[y][x] == b"|":
                    self.window.blit(self.median_vert[1], cell)
                elif desc[y][x] == b"-" and (x == 0 or desc[y][x - 1] != b"-"):
                    self.window.blit(self.median_horiz[0], cell)
                elif desc[y][x] == b"-" and (
                    x == desc.shape[1] - 1 or desc[y][x + 1] != b"-"
                ):
                    self.window.blit(self.median_horiz[2], cell)
                elif desc[y][x] == b"-":
                    self.window.blit(self.median_horiz[1], cell)

        for cell, color in zip(self.locs, self.locs_colors):
            color_cell = pygame.Surface(self.cell_size)
            color_cell.set_alpha(128)
            color_cell.fill(color)
            loc = self.get_surf_loc(cell)
            self.window.blit(color_cell, (loc[0], loc[1] + 10))
        taxiA_row,taxiA_col,taxiB_row,taxiB_col, passA_loc, destA_idx,passB_loc,destB_idx = self.decode(self.s)

        if self.lastaction != None and self.lastaction[0] in [0, 1, 2, 3]:
            self.taxi_orientation = self.lastaction[0]
        
        if self.lastaction != None and self.lastaction[1] in [0, 1, 2, 3]:
            self.taxiB_orientation = self.lastaction[1]
        
        destA_loc = self.get_surf_loc(self.locs[destA_idx])
        taxiA_location = self.get_surf_loc((taxiA_row, taxiA_col))
        destB_loc = self.get_surf_loc(self.locs[destB_idx])
        taxiB_location = self.get_surf_loc((taxiB_row, taxiB_col))
        
        self.window.blit(
            self.destination_img,
            (destA_loc[0], destA_loc[1] - self.cell_size[1] // 2),
        )
        self.window.blit(
            self.destinationB_img,
            (destB_loc[0], destB_loc[1] - self.cell_size[1] // 2),
        )
        
        if passA_loc < 4:
            self.window.blit(self.passenger_img, self.get_surf_loc(self.locs[passA_loc]))
            
        if passB_loc < 4:
            self.window.blit(self.passengerB_img, self.get_surf_loc(self.locs[passB_loc]))
        
        self.window.blit(self.taxiB_imgs[self.taxiB_orientation], taxiB_location)
        self.window.blit(self.taxi_imgs[self.taxi_orientation], taxiA_location)        

        if mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )

    def get_surf_loc(self, map_loc):
        return (map_loc[1] * 2 + 1) * self.cell_size[0], (
            map_loc[0] + 1
        ) * self.cell_size[1]

    def _render_text(self):
        desc = self.desc.copy().tolist()
        outfile = StringIO()

        out = [[c.decode("utf-8") for c in line] for line in desc]
        taxiA_row,taxiA_col,taxiB_row,taxiB_col, passA_loc, destA_idx,passB_loc,destB_idx = self.decode(self.s)

        def ul(x):
            return "_" if x == " " else x
        
        out[1 + taxiA_row][2 * taxiA_col + 1] = utils.colorize(
                out[1 + taxiA_row][2 * taxiA_col + 1], "gray", highlight=True
            )
        out[1 + taxiB_row][2 * taxiB_col + 1] = utils.colorize(
                ul(out[1 + taxiB_row][2 * taxiB_col + 1]), "yellow", highlight=True
            )
        if passA_loc < 4:
            pi, pj = self.locs[passA_loc]
            out[1 + pi][2 * pj + 1] = "1"
        elif passA_loc == 4:
            out[1 + taxiA_row][2 * taxiA_col + 1] = utils.colorize(
                out[1 + taxiA_row][2 * taxiA_col + 1], "green", highlight=True
            )
        else:  # passenger in taxi
            out[1 + taxiB_row][2 * taxiB_col + 1] = utils.colorize(
                ul(out[1 + taxiB_row][2 * taxiB_col + 1]), "blue", highlight=True
            )
        
        if passB_loc < 4:
            pi, pj = self.locs[passB_loc]
            out[1 + pi][2 * pj + 1] = "2"
        elif passB_loc == 4:
            out[1 + taxiA_row][2 * taxiA_col + 1] = utils.colorize(
                out[1 + taxiA_row][2 * taxiA_col + 1], "green", highlight=True
            )
        else:  # passenger in taxi
            out[1 + taxiB_row][2 * taxiB_col + 1] = utils.colorize(
                ul(out[1 + taxiB_row][2 * taxiB_col + 1]), "blue", highlight=True
            )

        di, dj = self.locs[destA_idx]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], "magenta")
        di, dj = self.locs[destB_idx]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], "green")
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write(
                f"  ({['South|South','South|North','South|East','South|West','South|Pickup','South|Dropoff',  'North|South','North|North','North|East','North|West','North|Pickup','North|Dropoff', 'East|South','East|North','East|East','East|West','East|Pickup','East|Dropoff', 'West|South','West|North','West|East','West|West','West|Pickup','West|Dropoff', 'Pickup|South','Pickup|North','Pickup|East','Pickup|West','Pickup|Pickup','Pickup|Dropoff','Dropoff|South','Dropoff|North','Dropoff|East','Dropoff|West','Dropoff|Pickup','Dropoff|Dropoff'][self.lastaction[0]*6+self.lastaction[1]]})\n"
            )
        else:
            outfile.write("\n")

        with closing(outfile):
            return outfile.getvalue()

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()


# Taxi rider from https://franuka.itch.io/rpg-asset-pack
# All other assets by Mel Tillery http://www.cyaneus.com/





