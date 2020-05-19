from gym_tsxoa.envs.probs import PROBLEMS
from gym_tsxoa.envs.reps import REPRESENTATIONS
import numpy as np
import PIL

"""
The PCGRL GYM Environment
"""
class PcgrlEnv:

    """
    Constructor for the interface.

    Parameters:
        prob (string): the current problem. This name has to be defined in PROBLEMS
        constant in gym_tsxoa.envs.probs.__init__.py file
        rep (string): the current representation. This name has to be defined in REPRESENTATIONS
        constant in gym_tsxoa.envs.reps.__init__.py
    """
    def __init__(self, prob="binary", rep="narrow", max_percentage=1.0):
        self._prob = PROBLEMS[prob]()
        self._rep = REPRESENTATIONS[rep]()
        self._rep_stats = None
        self._start_stats = None
        self._iteration = 0
        self._changes = 0
        self._max_changes = max(int(max_percentage * self._prob._width * self._prob._height), 1)
        self._max_iterations = self._max_changes * self._prob._width * self._prob._height
        self.seed()

    """
    Seeding the used random variable to get the same result. If the seed is None,
    it will seed it with random start.

    Parameters:
        seed (int): the starting seed, if it is None a random seed number is used.
    """
    def seed(self, seed=None):
        self._rep.seed(seed)

    """
    Resets the environment to the start state

    Returns:
        Observation: the current starting observation have structure defined by
        the Observation Space
    """
    def reset(self):
        self._changes = 0
        self._iteration = 0
        self._rep.reset(self._prob._width, self._prob._height, self._prob._prob)
        if self._start_stats == None:
            self._start_stats = self._prob.get_stats(self._rep._map)
        self._rep_stats = self._prob.get_stats(self._rep._map)

        obs = self.get_observation()
        heuristic = self._prob.get_heuristic(self._rep_stats, self._start_stats)
        game_done = self._prob.get_episode_over(self._rep_stats,self._start_stats)
        done = game_done or self._changes >= self._max_changes or self._iteration >= self._max_iterations
        info = self._prob.get_debug_info(self._rep_stats,self._start_stats)
        info["iterations"] = self._iteration
        info["changes"] = self._changes
        info["max_iterations"] = self._max_iterations
        info["max_changes"] = self._max_changes
        return obs, heuristic, game_done, done, info

    def get_number_action(self):
        return self._rep.get_number_action(self._prob._width, self._prob._height, len(self._prob.get_tile_types()))

    def get_state_key(self):
        return self._rep.get_state_key()

    def get_observation(self):
        obs = self._rep.get_observation()
        obs['changes'] = self._changes
        obs['iteration'] = self._iteration
        obs['rep_stats'] = self._rep_stats
        return obs

    def set_observation(self, obs, copy=True):
        self._rep.set_observation(obs, copy)
        self._changes = obs['changes']
        self._iteration = obs['iteration']
        self._rep_stats = obs['rep_stats']

    """
    Advance the environment using a specific action

    Parameters:
        action: an action that is used to advance the environment (same as action space)

    Returns:
        observation: the current observation after applying the action
        float: the reward that happened because of applying that action
        boolean: if the problem eneded (episode is over)
        dictionary: debug information that might be useful to understand what's happening
    """
    def step(self, action, earlyTermination=True, quick=False):
        if earlyTermination:
            self._iteration += 1
        # update the current state to the new state based on the taken action
        change, x, y = self._rep.update(action)
        if change > 0:
            if earlyTermination:
                self._changes += change
            earlyDone = self._changes >= self._max_changes or self._iteration >= self._max_iterations
            if quick and not earlyDone:
                return None, 0, False, False, {}
            self._rep_stats = self._prob.get_stats(self._rep._map)
        earlyDone = self._changes >= self._max_changes or self._iteration >= self._max_iterations
        if quick and not earlyDone:
            return None, 0, False, False, {}
        # calculate the values
        obs = self.get_observation()
        heuristic = self._prob.get_heuristic(self._rep_stats, self._start_stats)
        game_done = self._prob.get_episode_over(self._rep_stats,self._start_stats)
        done = game_done or self._changes >= self._max_changes or self._iteration >= self._max_iterations
        info = self._prob.get_debug_info(self._rep_stats,self._start_stats)
        info["iterations"] = self._iteration
        info["changes"] = self._changes
        info["max_iterations"] = self._max_iterations
        info["max_changes"] = self._max_changes
        #return the values
        return obs, heuristic, game_done, done, info

    def calculate_step(self):
        self._rep_stats = self._prob.get_stats(self._rep._map)
        obs = self.get_observation()
        heuristic = self._prob.get_heuristic(self._rep_stats, self._start_stats)
        game_done = self._prob.get_episode_over(self._rep_stats,self._start_stats)
        done = game_done or self._changes >= self._max_changes or self._iteration >= self._max_iterations
        info = self._prob.get_debug_info(self._rep_stats,self._start_stats)
        info["iterations"] = self._iteration
        info["changes"] = self._changes
        info["max_iterations"] = self._max_iterations
        info["max_changes"] = self._max_changes
        #return the values
        return obs, heuristic, game_done, done, info

    """
    Render the current state of the environment

    Parameters:
        mode (string): the value has to be defined in render.modes in metadata

    Returns:
        img or boolean: img for rgb_array rendering and boolean for human rendering
    """
    def render(self):
        tile_size=16
        img = self._prob.render(self._rep._map)
        img = self._rep.render(img, self._prob._tile_size, self._prob._border_size).convert("RGB")
        return img
