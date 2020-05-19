import os
import numpy as np
from PIL import Image
from gym_tsxoa.envs.probs.problem import Problem
from gym_tsxoa.envs.helper import get_range_reward, get_tile_locations, calc_num_regions, calc_longest_path

"""
Generate a fully connected top down layout where the longest path is greater than a certain threshold
"""
class BinaryProblem(Problem):
    """
    The constructor is responsible of initializing all the game parameters
    """
    def __init__(self):
        super().__init__()
        self._width = 14
        self._height = 14
        self._prob = {0: 0.5, 1:0.5}
        self._border_tile = 1

        self._target_path = 20

        self._rewards = {
            "regions": 5,
            "path-length": 1
        }

    """
    Get a list of all the different tile names

    Returns:
        string[]: that contains all the tile names
    """
    def get_tile_types(self):
        return [0, 1]

    """
    Get the current stats of the map

    Returns:
        dict(string,any): stats of the current map to be used in the reward, episode_over, debug_info calculations.
        The used status are "reigons": number of connected empty tiles, "path-length": the longest path across the map
    """
    def get_stats(self, map):
        map_locations = get_tile_locations(map, self.get_tile_types())
        return {
            "regions": calc_num_regions(map, map_locations, [0]),
            "path-length": calc_longest_path(map, map_locations, [0])
        }

    """
    Get the current game reward between two stats

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        start_stats (dict(string,any)): the old stats before taking an action

    Returns:
        float: the current reward due to the change between the old map stats and the new map stats
    """
    def get_heuristic(self, new_stats, start_stats):
        #longer path is rewarded and less number of regions is rewarded
        rewards = {
            "regions": get_range_reward(new_stats["regions"], start_stats["regions"], 1, 1),
            "path-length": get_range_reward(new_stats["path-length"],start_stats["path-length"], np.inf, np.inf)
        }
        #calculate the total reward
        return rewards["regions"] * self._rewards["regions"] +\
            rewards["path-length"] * self._rewards["path-length"]

    """
    Uses the stats to check if the problem ended (episode_over) which means reached
    a satisfying quality based on the stats

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        start_stats (dict(string,any)): the old stats before taking an action

    Returns:
        boolean: True if the level reached satisfying quality based on the stats and False otherwise
    """
    def get_episode_over(self, new_stats, start_stats):
        return new_stats["regions"] == 1 and new_stats["path-length"] - start_stats["path-length"] >= self._target_path

    """
    Get any debug information need to be printed

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        start_stats (dict(string,any)): the old stats before taking an action

    Returns:
        dict(any,any): is a debug information that can be used to debug what is
        happening in the problem
    """
    def get_debug_info(self, new_stats, start_stats):
        return {
            "regions": new_stats["regions"],
            "path-length": new_stats["path-length"],
            "path-imp": new_stats["path-length"] - start_stats["path-length"]
        }

    """
    Get an image on how the map will look like for a specific map

    Parameters:
        map (string[][]): the current game map

    Returns:
        Image: a pillow image on how the map will look like using the binary graphics
    """
    def render(self, map):
        if self._graphics == None:
            self._graphics = {
                0: Image.open(os.path.dirname(__file__) + "/binary/empty.png").convert('RGBA'),
                1: Image.open(os.path.dirname(__file__) + "/binary/solid.png").convert('RGBA')
            }
        return super().render(map)
