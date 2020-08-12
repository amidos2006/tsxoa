from gym_tsxoa.envs.reps.representation import Representation
from PIL import Image
import numpy as np

"""
The narrow representation where the agent is trying to modify the tile value of a certain
selected position that is selected randomly or sequentially similar to cellular automata
"""
class NarrowRepresentation(Representation):
    """
    Initialize all the parameters used by that representation
    """
    def __init__(self):
        super().__init__()

    """
    Resets the current representation where it resets the parent and the current
    modified location

    Parameters:
        width (int): the generated map width
        height (int): the generated map height
        prob (dict(int,float)): the probability distribution of each tile value
    """
    def reset(self, width, height, prob):
        super().reset(width, height, prob)
        self._tiles = [];
        for x in range(width):
            for y in range(height):
                self._tiles.append((x, y))
        self._random.shuffle(self._tiles)
        self._index = 0

    """
    Get the current representation observation object at the current moment

    Returns:
        observation: the current observation at the current moment. "pos" Integer
        x,y position for the current location. "map" 2D array of tile numbers
    """
    def get_observation(self):
        return {
            "index": self._index,
            "map": self._map
        }

    def set_observation(self, obs, copy=True):
        self._map = obs['map']
        if copy:
            self._map = obs['map'].copy()
        self._index = obs['index']

    def get_number_action(self, width, height, num_tiles):
        return num_tiles

    def get_state_key(self):
        map_size = self._map.shape[1] * self._map.shape[0]
        key = np.array2string(self._map.reshape(map_size,),separator="", max_line_width=map_size+1)[1:-1]
        (x, y) = self._tiles[self._index % len(self._tiles)]
        return '{}_{}_{}'.format(key, x, y)

    """
    Update the narrow representation with the input action

    Parameters:
        action: an action that is used to advance the environment (same as action space)

    Returns:
        boolean: True if the action change the map, False if nothing changed
    """
    def update(self, action):
        change = 0
        (x, y) = self._tiles[self._index % len(self._tiles)]
        change += [0,1][self._map[y][x] != action]
        self._map[y][x] = action
        self._index += 1
        return change, x, y

    """
    Modify the level image with a red rectangle around the tile that is
    going to be modified

    Parameters:
        lvl_image (img): the current level_image without modifications
        tile_size (int): the size of tiles in pixels used in the lvl_image
        border_size ((int,int)): an offeset in tiles if the borders are not part of the level

    Returns:
        img: the modified level image
    """
    def render(self, lvl_image, tile_size, border_size):
        x_graphics = Image.new("RGBA", (tile_size,tile_size), (0,0,0,0))
        for x in range(tile_size):
            x_graphics.putpixel((0,x),(255,0,0,255))
            x_graphics.putpixel((1,x),(255,0,0,255))
            x_graphics.putpixel((tile_size-2,x),(255,0,0,255))
            x_graphics.putpixel((tile_size-1,x),(255,0,0,255))
        for y in range(tile_size):
            x_graphics.putpixel((y,0),(255,0,0,255))
            x_graphics.putpixel((y,1),(255,0,0,255))
            x_graphics.putpixel((y,tile_size-2),(255,0,0,255))
            x_graphics.putpixel((y,tile_size-1),(255,0,0,255))
        (x, y) = self._tiles[self._index % len(self._tiles)]
        lvl_image.paste(x_graphics, ((x+border_size[0])*tile_size, (y+border_size[1])*tile_size,
                                        (x+border_size[0]+1)*tile_size,(y+border_size[1]+1)*tile_size), x_graphics)
        return lvl_image
