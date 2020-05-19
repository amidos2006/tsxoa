import os
import numpy as np
from PIL import Image
from gym_tsxoa.envs.probs.problem import Problem
from gym_tsxoa.envs.helper import get_range_reward, get_tile_locations, calc_num_regions, calc_certain_tile, run_dikjstra

"""
Generate a fully connected GVGAI zelda level where the player can reach key then the door.

Args:
    target_enemy_dist: enemies should be at least this far from the player on spawn
"""
class ZeldaProblem(Problem):
    """
    The constructor is responsible of initializing all the game parameters
    """
    def __init__(self):
        super().__init__()
        self._width = 11
        self._height = 7
        self._prob = {0: 0.58, 1:0.3, 2:0.02, 3: 0.02, 4: 0.02, 5: 0.02, 6: 0.02, 7: 0.02}
        self._border_tile = 1

        self._max_enemies = 5

        self._target_enemy_dist = 4
        self._target_path = 16

        self._rewards = {
            "player": 3,
            "key": 3,
            "door": 3,
            "regions": 5,
            "enemies": 1,
            "nearest-enemy": 2,
            "path-length": 1
        }

    """
    Get a list of all the different tile names

    Returns:
        string[]: that contains all the tile names
    """
    def get_tile_types(self):
        return [0, 1, 2, 3, 4, 5, 6, 7]

    """
    Get the current stats of the map

    Returns:
        dict(string,any): stats of the current map to be used in the reward, episode_over, debug_info calculations.
        The used status are "reigons": number of connected empty tiles, "path-length": the longest path across the map
    """
    def get_stats(self, map):
        map_locations = get_tile_locations(map, self.get_tile_types())
        map_stats = {
            "player": calc_certain_tile(map_locations, [2]),
            "key": calc_certain_tile(map_locations, [3]),
            "door": calc_certain_tile(map_locations, [4]),
            "enemies": calc_certain_tile(map_locations, [5, 6, 7]),
            "regions": calc_num_regions(map, map_locations, [0, 2, 3, 5, 6, 7]),
            "nearest-enemy": 0,
            "path-length": 0
        }
        if map_stats["player"] == 1 and map_stats["regions"] == 1:
            p_x,p_y = map_locations[2][0]
            enemies = []
            enemies.extend(map_locations[5])
            enemies.extend(map_locations[6])
            enemies.extend(map_locations[7])
            if len(enemies) > 0:
                dikjstra,_ = run_dikjstra(p_x, p_y, map, [0, 2, 5, 6, 7])
                min_dist = self._width * self._height
                for e_x,e_y in enemies:
                    if dikjstra[e_y][e_x] > 0 and dikjstra[e_y][e_x] < min_dist:
                        min_dist = dikjstra[e_y][e_x]
                map_stats["nearest-enemy"] = min_dist
            if map_stats["key"] == 1 and map_stats["door"] == 1:
                k_x,k_y = map_locations[3][0]
                d_x,d_y = map_locations[4][0]
                dikjstra,_ = run_dikjstra(p_x, p_y, map, [0, 3, 2, 5, 6, 7])
                map_stats["path-length"] += dikjstra[k_y][k_x]
                dikjstra,_ = run_dikjstra(k_x, k_y, map, [0, 2, 3, 4, 5, 6, 7])
                map_stats["path-length"] += dikjstra[d_y][d_x]

        return map_stats

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
            "player": get_range_reward(new_stats["player"], start_stats["player"], 1, 1),
            "key": get_range_reward(new_stats["key"], start_stats["key"], 1, 1),
            "door": get_range_reward(new_stats["door"], start_stats["door"], 1, 1),
            "enemies": get_range_reward(new_stats["enemies"], start_stats["enemies"], 2, self._max_enemies),
            "regions": get_range_reward(new_stats["regions"], start_stats["regions"], 1, 1),
            "nearest-enemy": get_range_reward(new_stats["nearest-enemy"], start_stats["nearest-enemy"], self._target_enemy_dist, np.inf),
            "path-length": get_range_reward(new_stats["path-length"],start_stats["path-length"], np.inf, np.inf)
        }
        #calculate the total reward
        return rewards["player"] * self._rewards["player"] +\
            rewards["key"] * self._rewards["key"] +\
            rewards["door"] * self._rewards["door"] +\
            rewards["enemies"] * self._rewards["enemies"] +\
            rewards["regions"] * self._rewards["regions"] +\
            rewards["nearest-enemy"] * self._rewards["nearest-enemy"] +\
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
        return new_stats["nearest-enemy"] >= self._target_enemy_dist and new_stats["path-length"] >= self._target_path

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
            "player": new_stats["player"],
            "key": new_stats["key"],
            "door": new_stats["door"],
            "enemies": new_stats["enemies"],
            "regions": new_stats["regions"],
            "nearest-enemy": new_stats["nearest-enemy"],
            "path-length": new_stats["path-length"]
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
                0: Image.open(os.path.dirname(__file__) + "/zelda/empty.png").convert('RGBA'),
                1: Image.open(os.path.dirname(__file__) + "/zelda/solid.png").convert('RGBA'),
                2: Image.open(os.path.dirname(__file__) + "/zelda/player.png").convert('RGBA'),
                3: Image.open(os.path.dirname(__file__) + "/zelda/key.png").convert('RGBA'),
                4: Image.open(os.path.dirname(__file__) + "/zelda/door.png").convert('RGBA'),
                5: Image.open(os.path.dirname(__file__) + "/zelda/spider.png").convert('RGBA'),
                6: Image.open(os.path.dirname(__file__) + "/zelda/bat.png").convert('RGBA'),
                7: Image.open(os.path.dirname(__file__) + "/zelda/scorpion.png").convert('RGBA'),
            }
        return super().render(map)
