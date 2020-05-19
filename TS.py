import numpy as np
import random
from queue import PriorityQueue
import time
import math

class Node:
    def __init__(self, parent):
        self.parent = parent
        self.depth = 0
        if parent is not None:
            self.depth = self.parent.depth + 1
        self.heuristic = 0
        self.leaf = False
        self.win = False
        self.obs = None

    def random_init(self, env):
        obs, heuristic, game_done, done, info = env.reset()
        self.obs = obs
        self.heuristic = heuristic
        self.leaf = done
        self.win = game_done

    def expand_children(self, env):
        if self.leaf:
            return []
        number = env.get_number_action()
        children = []
        for i in range(number):
            n = Node(self)
            env.set_observation(self.obs)
            obs, heuristic, game_done, done, info = env.step(i)
            n.heuristic = heuristic
            n.win = game_done
            n.leaf = done
            n.obs = obs
            children.append(n)
        return children

    def get_key(self,env):
        env.set_observation(self.obs, False)
        return env.get_state_key()

    def get_heuristic(self):
        return self.heuristic

    def __str__(self):
        result = ""
        for y in range(self.obs['map'].shape[0]):
            for x in range(self.obs['map'].shape[1]):
                result += str(self.obs['map'][y][x])
            result += "\n"
        return result[:-1]

    def __lt__(self, other):
        return self.get_heuristic() > other.get_heuristic()

class MCTSNode:
    def __init__(self, parent):
        self.parent = parent
        self.win = False
        self.leaf = False
        self.children = []
        self.possible_children = None
        self.depth = 0
        if parent is not None:
            self.depth = self.parent.depth + 1
        self.total_value = 0
        self.total_visits = 0
        self.heuristic = 0
        self.obs = None

    def expand_possible_children(self, env, visited):
        self.possible_children = []
        actions = env.get_number_action()
        for a in range(actions):
            child = MCTSNode(self)
            env.set_observation(self.obs)
            obs, heuristic, game_done, done, info = env.step(a)
            child.obs = obs
            child.heuristic = heuristic
            child.win = game_done
            child.leaf = done
            key = child.get_key(env)
            if key not in visited:
                visited.add(key)
                self.possible_children.append(child)

    def random_init(self, env):
        obs, heuristic, game_done, done, info = env.reset()
        self.obs = obs
        self.heuristic = heuristic
        self.leaf = done
        self.game_done = game_done

    def get_ucb(self, c=1):
        return self.total_value / self.total_visits + c * math.sqrt((2*math.log(self.parent.total_visits))/self.total_visits)

    def get_heuristic(self):
        return self.heuristic

    def select(self, c=1):
        current = self
        while current.fully_expanded() and not current.terminal():
            max_ucb = -1
            max_child = None
            for child in current.children:
                current_ucb = child.get_ucb(c)
                if max_ucb == -1 or max_ucb < current_ucb:
                    max_ucb = current_ucb
                    max_child = child
            current = max_child
        return current

    def expand(self):
        if len(self.possible_children) > 0:
            child = self.possible_children.pop(np.random.randint(len(self.possible_children)))
            self.children.append(child)
            return child
        return self

    def fully_expanded(self):
        return self.possible_children is not None and len(self.possible_children) == 0

    def terminal(self):
        return self.leaf or (self.fully_expanded() and len(self.children) == 0)

    def simulate(self, env, length):
        env.set_observation(self.obs)
        actions = env.get_number_action()
        heuristic = 0
        for i in range(length):
            quick = i < length - 1
            obs, heuristic, game_done, done, info = env.step(env._rep._random.integers(actions), True, quick)
            if done:
                break
        return heuristic

    def backpropagate(self, value):
        self.total_value += value
        self.total_visits += 1
        if self.parent is not None:
            self.parent.backpropagate(value)

    def get_key(self, env):
        env.set_observation(self.obs, False)
        return env.get_state_key()

    def __str__(self):
        result = ""
        for y in range(self.obs['map'].shape[0]):
            for x in range(self.obs['map'].shape[1]):
                result += str(self.obs['map'][y][x])
            result += "\n"
        return result[:-1]

class SpecialMCTSNode:
    def __init__(self, parent):
        self.parent = parent
        self.win = False
        self.leaf = False
        self.children = []
        self.possible_children = None
        self.depth = 0
        if parent is not None:
            self.depth = self.parent.depth + 1
        self.total_value = 0
        self.total_visits = 0
        self.min_value = -1
        self.max_value = -1
        self.heuristic = 0
        self.obs = None

    def expand_possible_children(self, env, visited):
        self.possible_children = []
        actions = env.get_number_action()
        for a in range(actions):
            child = SpecialMCTSNode(self)
            env.set_observation(self.obs)
            obs, heuristic, game_done, done, info = env.step(a)
            child.obs = obs
            child.heuristic = heuristic
            child.win = game_done
            child.leaf = done
            key = child.get_key(env)
            if key not in visited:
                visited.add(key)
                self.possible_children.append(child)

    def random_init(self, env):
        obs, heuristic, game_done, done, info = env.reset()
        self.obs = obs
        self.heuristic = heuristic
        self.leaf = done
        self.game_done = game_done

    def get_ucb(self, addedConst=0, weightConst = 1):
        c = addedConst + weightConst * (self.parent.max_value - self.parent.min_value + 1e-6)
        return self.total_value / self.total_visits + c * math.sqrt((2*math.log(self.parent.total_visits))/self.total_visits)

    def get_heuristic(self):
        return self.heuristic

    def select(self, addedConst=0, weightConst = 1):
        current = self
        while current.fully_expanded() and not current.terminal():
            max_ucb = -1
            max_child = None
            for c in current.children:
                current_ucb = c.get_ucb(addedConst, weightConst)
                if max_ucb == -1 or max_ucb < current_ucb:
                    max_ucb = current_ucb
                    max_child = c
            current = max_child
        return current

    def expand(self):
        if len(self.possible_children) > 0:
            child = self.possible_children.pop(np.random.randint(len(self.possible_children)))
            self.children.append(child)
            return child
        return self

    def fully_expanded(self):
        return self.possible_children is not None and len(self.possible_children) == 0

    def terminal(self):
        return self.leaf or (self.fully_expanded() and len(self.children) == 0)

    def simulate(self, env, length):
        env.set_observation(self.obs)
        actions = env.get_number_action()
        heuristic = 0
        for i in range(length):
            quick = i < length - 1
            obs, heuristic, game_done, done, info = env.step(env._rep._random.integers(actions), True, quick)
            if done:
                break
        return heuristic

    def backpropagate(self, value, new_value):
        self.total_value += value
        self.total_visits += 1
        if self.fully_expanded():
            if self.min_value == -1 or self.min_value > new_value:
                self.min_value = new_value
            if self.max_value == -1 or self.max_value < new_value:
                self.max_value = new_value
        if self.parent is not None:
            self.parent.backpropagate(value, self.total_value / self.total_visits)

    def get_key(self, env):
        env.set_observation(self.obs, False)
        return env.get_state_key()

    def __str__(self):
        result = ""
        for y in range(self.obs['map'].shape[0]):
            for x in range(self.obs['map'].shape[1]):
                result += str(self.obs['map'][y][x])
            result += "\n"
        return result[:-1]

class TS:
    def __init__(self, env):
        self.root = Node(None)
        self.root.random_init(env)
        self.best_node = self.root
        self.deep_node = self.root

    def run(self, env, maxTime=60):
        self.checked_nodes = 0
        self.time_out = maxTime

    def get_best(self):
        return self.best_node

    def get_deep(self):
        return self.deep_node

class BFS(TS):
    def __init__(self, env):
        super().__init__(env)

    def run(self, env, maxTime=60):
        super().run(env, maxTime)
        visited = set()

        queue = [self.root]
        start_time = time.time()
        while time.time() - start_time < maxTime and len(queue) > 0:
            current = queue.pop(0)
            self.checked_nodes += 1
            if current.get_key(env) not in visited:
                if current.get_heuristic() > self.best_node.get_heuristic():
                    self.best_node = current
                elif current.get_heuristic() == self.best_node.get_heuristic() and np.random.random() < 0.5:
                    self.best_node = current
                if current.depth > self.deep_node.depth:
                    self.deep_node = current
                if current.win:
                    self.best_node = current
                    self.time_out = time.time() - start_time
                    return
                visited.add(current.get_key(env))
                queue.extend(current.expand_children(env))

class DFS(TS):
    def __init__(self, env):
        super().__init__(env)

    def run(self, env, maxTime=60):
        super().run(env, maxTime)
        visited = set()

        queue = [self.root]
        start_time = time.time()
        while time.time() - start_time < maxTime and len(queue) > 0:
            current = queue.pop()
            self.checked_nodes += 1
            if current.get_key(env) not in visited:
                if current.get_heuristic() > self.best_node.get_heuristic():
                    self.best_node = current
                elif current.get_heuristic() == self.best_node.get_heuristic() and np.random.random() < 0.5:
                    self.best_node = current
                if current.depth > self.deep_node.depth:
                    self.deep_node = current
                if current.win:
                    self.best_node = current
                    self.time_out = time.time() - start_time
                    return
                visited.add(current.get_key(env))
                queue.extend(current.expand_children(env))

class BestFS(TS):
    def __init__(self, env):
        super().__init__(env)

    def run(self, env, maxTime=60):
        super().run(env, maxTime)
        visited = set()

        queue = PriorityQueue()
        queue.put(self.root)
        start_time = time.time()
        while time.time() - start_time < maxTime and queue.qsize() > 0:
            current = queue.get()
            self.checked_nodes += 1
            if current.get_key(env) not in visited:
                if current.get_heuristic() > self.best_node.get_heuristic():
                    self.best_node = current
                elif current.get_heuristic() == self.best_node.get_heuristic() and np.random.random() < 0.5:
                    self.best_node = current
                if current.depth > self.deep_node.depth:
                    self.deep_node = current
                if current.win:
                    self.best_node = current
                    self.time_out = time.time() - start_time
                    return
                visited.add(current.get_key(env))
                children = current.expand_children(env)
                for c in children:
                    queue.put(c)

class SpecialMCTS:
    def __init__(self, env):
        self.root = SpecialMCTSNode(None)
        self.root.random_init(env)
        self.best_node = self.root
        self.deep_node = self.root
        self.time_out = 0
        self.checked_nodes = 0

    def run(self, env, maxTime=60, rollout=10, addedC=0, multC=1):
        visited = set()
        if self.root.win:
            return
        self.time_out = maxTime
        start_time = time.time()
        while time.time() - start_time < maxTime:
            current = self.root.select(addedC, multC)
            if not current.terminal():
                if current.possible_children == None:
                    self.checked_nodes += 1
                    current.expand_possible_children(env, visited)
                current = current.expand()
            if current.get_heuristic() > self.best_node.get_heuristic():
                self.best_node = current
            if current.depth > self.deep_node.depth:
                self.deep_node = current
            if current.win:
                self.best_node = current
                self.time_out = time.time() - start_time
                return
            value = current.simulate(env, rollout)
            current.backpropagate(value, (current.total_value+value) / (current.total_visits+1))

class MCTS:
    def __init__(self, env):
        self.root = MCTSNode(None)
        self.root.random_init(env)
        self.best_node = self.root
        self.deep_node = self.root
        self.time_out = 0
        self.checked_nodes = 0

    def run(self, env, maxTime=60, c=1, rollout=10):
        visited = set()
        if self.root.win:
            return
        self.time_out = maxTime
        start_time = time.time()
        while time.time() - start_time < maxTime:
            current = self.root.select(c)
            if not current.terminal():
                if current.possible_children == None:
                    self.checked_nodes += 1
                    current.expand_possible_children(env, visited)
                current = current.expand()
            if current.get_heuristic() > self.best_node.get_heuristic():
                self.best_node = current
            if current.depth > self.deep_node.depth:
                self.deep_node = current
            if current.win:
                self.best_node = current
                self.time_out = time.time() - start_time
                return
            value = current.simulate(env, rollout)
            current.backpropagate(value)
