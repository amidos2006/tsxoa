from gym_tsxoa.envs.reps.representation import Representation
from PIL import Image
import numpy as np

"""
The wide representation where the agent can pick the tile position and tile value at each update.
"""
class WideRepresentation(Representation):
    """
    Initialize all the parameters used by that representation
    """
    def __init__(self):
        super().__init__()

    """
    Get the current representation observation object at the current moment

    Returns:
        observation: the current observation at the current moment. A 2D array of tile numbers
    """
    def get_observation(self):
        return {
            "map": self._map
        }

    def set_observation(self, obs, copy=True):
        self._map = obs['map']
        if copy:
            self._map = obs['map'].copy()

    def get_number_action(self, width, height, num_tiles):
        return width * height * num_tiles

    def get_state_key(self):
        map_size = self._map.shape[1] * self._map.shape[0]
        key = np.array2string(self._map.reshape(map_size,),separator="", max_line_width=map_size+1)[1:-1]
        return key

    """
    Update the wide representation with the input action

    Parameters:
        action: an action that is used to advance the environment (same as action space)

    Returns:
        boolean: True if the action change the map, False if nothing changed
    """
    def update(self, action):
        x = action % self._map.shape[1]
        action = int(action / self._map.shape[1])
        y = action % self._map.shape[0]
        action = int(action / self._map.shape[0])
        value = action

        change = [0,1][self._map[y][x] != value]
        self._map[y][x] = value
        return change, x, y
