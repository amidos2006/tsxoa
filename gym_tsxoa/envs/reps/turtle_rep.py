from gym_tsxoa.envs.reps.representation import Representation
from PIL import Image
import numpy as np

"""
The turtle representation where the agent is trying to modify the position of the
turtle or the tile value of its current location similar to turtle graphics.
The difference with narrow representation is the agent now controls the next tile to be modified.
"""
class TurtleRepresentation(Representation):
    """
    Initialize all the parameters used by that representation
    """
    def __init__(self):
        super().__init__()
        self._dirs = [(-1,0), (1,0), (0,-1), (0,1)]

    """
    Resets the current representation where it resets the parent and the current
    turtle location

    Parameters:
        width (int): the generated map width
        height (int): the generated map height
        prob (dict(int,float)): the probability distribution of each tile value
    """
    def reset(self, width, height, prob):
        super().reset(width, height, prob)
        self._x = self._random.integers(width)
        self._y = self._random.integers(height)

    """
    Get the current representation observation object at the current moment

    Returns:
        observation: the current observation at the current moment. "pos" Integer
        x,y position for the current location. "map" 2D array of tile numbers
    """
    def get_observation(self):
        return {
            "x": self._x,
            "y": self._y,
            "map": self._map
        }

    def set_observation(self, obs, copy=True):
        self._map = obs['map']
        if copy:
            self._map = obs['map'].copy()
        self._x = obs['x']
        self._y = obs['y']

    def get_number_action(self, width, height, num_tiles):
        return len(self._dirs) + num_tiles

    def get_state_key(self):
        map_size = self._map.shape[1] * self._map.shape[0]
        key = np.array2string(self._map.reshape(map_size,),separator="", max_line_width=map_size+1)[1:-1]
        return '{}_{}_{}'.format(key, self._x, self._y)

    """
    Update the turtle representation with the input action

    Parameters:
        action: an action that is used to advance the environment (same as action space)

    Returns:
        boolean: True if the action change the map, False if nothing changed
    """
    def update(self, action):
        change = 0
        if action < len(self._dirs):
            self._x += self._dirs[action][0]
            if self._x < 0:
                self._x = 0
            if self._x >= self._map.shape[1]:
                self._x = self._map.shape[1] - 1
            self._y += self._dirs[action][1]
            if self._y < 0:
                self._y = 0
            if self._y >= self._map.shape[0]:
                self._y = self._map.shape[0] - 1
        else:
            change = [0,1][self._map[self._y][self._x] != action - len(self._dirs)]
            self._map[self._y][self._x] = action - len(self._dirs)
        return change, self._x, self._y

    """
    Modify the level image with a red rectangle around the tile that the turtle is on

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
        lvl_image.paste(x_graphics, ((self._x+border_size[0])*tile_size, (self._y+border_size[1])*tile_size,
                                        (self._x+border_size[0]+1)*tile_size,(self._y+border_size[1]+1)*tile_size), x_graphics)
        return lvl_image
