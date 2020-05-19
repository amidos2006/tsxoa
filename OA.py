import math
import time

class Chromosome:
    def __init__(self):
        self.obs = None
        self.fitness = -1
        self.win = False

    def random_init(self, env):
        obs, fitness, game_done, done, info = env.reset()
        self.obs = obs
        self.fitness = fitness
        self.win = game_done

    def copy(self):
        clone = Chromosome()
        clone.fitness = self.fitness
        clone.win = self.win
        clone.obs = {}
        for k in self.obs:
            clone.obs[k] = self.obs[k]
            if hasattr(self.obs[k], 'copy'):
                clone.obs[k] = clone.obs[k].copy()
        return clone

    def get_all_neighbors(self, env):
        neighbors = []
        actions = env.get_number_action()
        for a in range(actions):
            env.set_observation(self.obs)
            obs, fitness, game_done, done, info = env.step(a, False)
            c = Chromosome()
            c.fitness = fitness
            c.win = game_done
            c.obs = obs
            neighbors.append(c)
        return neighbors

    def get_neighbor(self, env):
        actions = env.get_number_action()
        env.set_observation(self.obs)
        obs, fitness, game_done, done, info = env.step(env._rep._random.integers(actions), False)
        c = Chromosome()
        c.fitness = fitness
        c.win = game_done
        c.obs = obs
        return c

    def crossover(self, env, other):
        c = self.copy()
        sx = env._rep._random.integers(self.obs['map'].shape[1])
        sy = env._rep._random.integers(self.obs['map'].shape[0])
        ex = max(1, env._rep._random.integers(sx + 1, self.obs['map'].shape[1] + 1))
        ey = max(1, env._rep._random.integers(sy + 1, self.obs['map'].shape[0] + 1))
        for y in range(sy, ey):
            for x in range(sx, ex):
                c.obs['map'][y][x] = other.obs['map'][y][x]
        env.set_observation(self.obs)
        obs, fitness, game_done, done, info = env.calculate_step()
        c.obs = obs
        c.fitness = fitness
        c.win = game_done
        return c

    def get_fitness(self):
        return self.fitness

    def __str__(self):
        result = ""
        for y in range(self.obs['map'].shape[0]):
            for x in range(self.obs['map'].shape[1]):
                result += str(self.obs['map'][y][x])
            result += "\n"
        return result[:-1]

class OA:
    def __init__(self, env):
        pass

    def advance(self, env):
        pass

    def run(self, env, maxTime=60):
        self.gen = 0
        start_time = time.time()
        while True:
            if time.time() - start_time >= maxTime or self.get_best().win:
                self.time_out = time.time() - start_time
                break
            self.advance(env)
            self.gen += 1

class HC(OA):
    def __init__(self, env):
        super().__init__(env)

        self._current = Chromosome()
        self._current.random_init(env)

    def advance(self, env):
        super().__init__(env)

        neighbors = self._current.get_all_neighbors(env)
        best = self._current
        for n in neighbors:
            if n.get_fitness() > best.get_fitness():
                best = n
            elif n.get_fitness() == best.get_fitness() and env._rep._random.random() < 0.5:
                best = n
        self._current = best

    def get_best(self):
        return self._current

class SA(OA):
    def __init__(self, env):
        super().__init__(env)

        self._cooling=0.99
        self._temp = 10
        self._current = Chromosome()
        self._current.random_init(env)

    def advance(self, env):
        super().__init__(env)

        neighbor = self._current.get_neighbor(env)
        d = self._current.get_fitness() - neighbor.get_fitness()
        if neighbor.get_fitness() > self._current.get_fitness():
            self._current = neighbor
        elif env._rep._random.random() < math.exp(-d / self._temp):
                self._current = neighbor
        self._temp = self._cooling * self._temp

    def get_best(self):
        return self._current

class ES(OA):
    def __init__(self, env):
        super().__init__(env)

        self._mu=10
        self._lambda = 20
        self._pop = []
        for i in range(self._mu + self._lambda):
            self._pop.append(Chromosome())
            self._pop[-1].random_init(env)
        self._pop = sorted(self._pop, key=lambda x: x.get_fitness(), reverse=True)

    def advance(self, env):
        super().__init__(env)

        for i in range(self._mu):
            self._pop[self._mu + 2*i] = self._pop[i].get_neighbor(env)
            self._pop[self._mu + 2*i + 1] = self._pop[i].get_neighbor(env)
        self._pop = sorted(self._pop, key=lambda x: x.get_fitness(), reverse=True)

    def get_best(self):
        return self._pop[0]

class GA(OA):
    def __init__(self, env):
        super().__init__(env)

        self._size=30
        self._crossover = 0.8
        self._elitism = 1
        self._mutation = 0.05
        self._pop = []
        for i in range(self._size):
            self._pop.append(Chromosome())
            self._pop[-1].random_init(env)
        self._pop = sorted(self._pop, key=lambda x: x.get_fitness(), reverse=True)

    def rank_select(self, env):
        indeces = list(range(1, self._size+1))
        indeces.reverse()
        for i in range(1,self._size):
            indeces[i] = indeces[i] + indeces[i-1]
        random = env._rep._random.random()
        for i in range(self._size):
            if random < indeces[i] / indeces[-1]:
                return i
        return self._size - 1

    def advance(self, env):
        new_pop = []
        for i in range(self._elitism):
            new_pop.append(self._pop[i].copy())
        while len(new_pop) < self._size:
            if env._rep._random.random() < self._crossover:
                parent1 = self._pop[self.rank_select(env)]
                parent2 = self._pop[self.rank_select(env)]
                child = parent1.crossover(env, parent2)
                if env._rep._random.random() < self._mutation:
                    child = child.get_neighbor(env)
                new_pop.append(child)
            else:
                parent = self._pop[self.rank_select(env)]
                new_pop.append(parent.get_neighbor(env))
        self._pop = sorted(new_pop, key=lambda x: x.get_fitness(), reverse=True)

    def get_best(self):
        return self._pop[0]
