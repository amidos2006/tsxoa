import os
from PIL import Image
import numpy as np
from gym_tsxoa.envs.probs.problem import Problem
from gym_tsxoa.envs.helper import get_range_reward, get_tile_locations, calc_certain_tile, calc_num_regions
from gym_tsxoa.envs.probs.sokoban.engine import State,BFSAgent,AStarAgent

"""
Generate a fully connected Sokoban(https://en.wikipedia.org/wiki/Sokoban) level that can be solved
"""
class SokobanProblem(Problem):
    """
    The constructor is responsible of initializing all the game parameters
    """
    def __init__(self):
        super().__init__()
        self._width = 5
        self._height = 5
        self._prob = {0:0.45, 1:0.4, 2: 0.05, 3: 0.05, 4: 0.05}
        self._border_tile = 1

        self._solver_power = 5000

        self._max_crates = 3

        self._target_solution = 18

        self._rewards = {
            "player": 3,
            "crate": 2,
            "target": 2,
            "regions": 5,
            "ratio": 2,
            "dist-win": 0.0,
            "sol-length": 1
        }

    """
    Get a list of all the different tile names

    Returns:
        string[]: that contains all the tile names
    """
    def get_tile_types(self):
        return [0, 1, 2, 3, 4]

    """
    Private function that runs the game on the input level

    Parameters:
        map (string[][]): the input level to run the game on

    Returns:
        float: how close you are to winning (0 if you win)
        int: the solution length if you win (0 otherwise)
    """
    def _run_game(self, map):
        gameCharacters=" #@$."
        string_to_char = dict((s, gameCharacters[i]) for i, s in enumerate(self.get_tile_types()))
        lvlString = ""
        for x in range(self._width+2):
            lvlString += "#"
        lvlString += "\n"
        for i in range(len(map)):
            for j in range(len(map[i])):
                string = map[i][j]
                if j == 0:
                    lvlString += "#"
                lvlString += string_to_char[string]
                if j == self._width-1:
                    lvlString += "#\n"
        for x in range(self._width+2):
            lvlString += "#"
        lvlString += "\n"

        state = State()
        state.stringInitialize(lvlString.split("\n"))

        aStarAgent = AStarAgent()
        bfsAgent = BFSAgent()

        sol,solState,iters = bfsAgent.getSolution(state, self._solver_power)
        if solState.checkWin():
            return 0, sol
        sol,solState,iters = aStarAgent.getSolution(state, 1, self._solver_power)
        if solState.checkWin():
            return 0, sol
        sol,solState,iters = aStarAgent.getSolution(state, 0.5, self._solver_power)
        if solState.checkWin():
            return 0, sol
        sol,solState,iters = aStarAgent.getSolution(state, 0, self._solver_power)
        if solState.checkWin():
            return 0, sol
        return solState.getHeuristic(), []

    """
    Get the current stats of the map

    Returns:
        dict(string,any): stats of the current map to be used in the reward, episode_over, debug_info calculations.
        The used status are "player": number of player tiles, "crate": number of crate tiles,
        "target": number of target tiles, "reigons": number of connected empty tiles,
        "dist-win": how close to the win state, "sol-length": length of the solution to win the level
    """
    def get_stats(self, map):
        map_locations = get_tile_locations(map, self.get_tile_types())
        map_stats = {
            "player": calc_certain_tile(map_locations, [2]),
            "crate": calc_certain_tile(map_locations, [3]),
            "target": calc_certain_tile(map_locations, [4]),
            "regions": calc_num_regions(map, map_locations, [0,2,3,4]),
            "dist-win": self._width * self._height * (self._width + self._height),
            "solution": []
        }
        if map_stats["player"] == 1 and map_stats["crate"] == map_stats["target"] and map_stats["crate"] > 0 and map_stats["regions"] == 1:
                map_stats["dist-win"], map_stats["solution"] = self._run_game(map)
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
            "crate": get_range_reward(new_stats["crate"], start_stats["crate"], 1, self._max_crates),
            "target": get_range_reward(new_stats["target"], start_stats["target"], 1, self._max_crates),
            "regions": get_range_reward(new_stats["regions"], start_stats["regions"], 1, 1),
            "ratio": get_range_reward(abs(new_stats["crate"]-new_stats["target"]), abs(start_stats["crate"]-start_stats["target"]), -np.inf, -np.inf),
            "dist-win": get_range_reward(new_stats["dist-win"], start_stats["dist-win"], -np.inf, -np.inf),
            "sol-length": get_range_reward(len(new_stats["solution"]), len(start_stats["solution"]), np.inf, np.inf)
        }
        #calculate the total reward
        return rewards["player"] * self._rewards["player"] +\
            rewards["crate"] * self._rewards["crate"] +\
            rewards["target"] * self._rewards["target"] +\
            rewards["regions"] * self._rewards["regions"] +\
            rewards["ratio"] * self._rewards["ratio"] +\
            rewards["dist-win"] * self._rewards["dist-win"] +\
            rewards["sol-length"] * self._rewards["sol-length"]

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
        return len(new_stats["solution"]) >= self._target_solution

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
            "crate": new_stats["crate"],
            "target": new_stats["target"],
            "regions": new_stats["regions"],
            "dist-win": new_stats["dist-win"],
            "sol-length": len(new_stats["solution"])
        }

    """
    Get an image on how the map will look like for a specific map

    Parameters:
        map (string[][]): the current game map

    Returns:
        Image: a pillow image on how the map will look like using sokoban graphics
    """
    def render(self, map):
        if self._graphics == None:
            self._graphics = {
                0: Image.open(os.path.dirname(__file__) + "/sokoban/empty.png").convert('RGBA'),
                1: Image.open(os.path.dirname(__file__) + "/sokoban/solid.png").convert('RGBA'),
                2: Image.open(os.path.dirname(__file__) + "/sokoban/player.png").convert('RGBA'),
                3: Image.open(os.path.dirname(__file__) + "/sokoban/crate.png").convert('RGBA'),
                4: Image.open(os.path.dirname(__file__) + "/sokoban/target.png").convert('RGBA')
            }
        return super().render(map)
