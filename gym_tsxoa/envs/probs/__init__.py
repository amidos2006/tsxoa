from gym_tsxoa.envs.probs.binary_prob import BinaryProblem
from gym_tsxoa.envs.probs.sokoban_prob import SokobanProblem
from gym_tsxoa.envs.probs.zelda_prob import ZeldaProblem

# all the problems should be defined here with its corresponding class
PROBLEMS = {
    "binary": BinaryProblem,
    "sokoban": SokobanProblem,
    "zelda": ZeldaProblem
}
