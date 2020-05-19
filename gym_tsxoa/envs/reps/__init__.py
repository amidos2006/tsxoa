from gym_tsxoa.envs.reps.narrow_rep import NarrowRepresentation
from gym_tsxoa.envs.reps.wide_rep import WideRepresentation
from gym_tsxoa.envs.reps.turtle_rep import TurtleRepresentation

# all the representations should be defined here with its corresponding class
REPRESENTATIONS = {
    "narrow": NarrowRepresentation,
    "wide": WideRepresentation,
    "turtle": TurtleRepresentation
}
