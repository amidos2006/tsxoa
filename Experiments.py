import OA
import TS
from gym_tsxoa.envs import PcgrlEnv
import sys
import os

algorithms = ["BFS", "DFS", "AStar", "MCTS", "HC", "SA", "ES", "GA"]
oa_start = 5
algorithms_dict = {
    "BFS": TS.BFS,
    "DFS": TS.DFS,
    "AStar": TS.BestFS,
    "MCTS": TS.MCTS,
    "DeepMCTS": TS.MCTS,
    "HC": OA.HC,
    "SA": OA.SA,
    "ES": OA.ES,
    "GA": OA.GA
}

problems = ["binary", "zelda", "sokoban"]
representations = ["narrow", "turtle", "wide"]
c_value = {
    'binary': 5,
    'zelda': 5,
    'sokoban': 5
}
roll_value = {
    'binary': 78,
    'zelda': 40,
    'sokoban': 10
}

experiments = []
for algo in algorithms:
    for prob in problems:
        for rep in representations:
            if algorithms.index(algo) >= oa_start and rep != "wide":
                continue
            experiments.append({
                "algo": algo,
                "prob": prob,
                "rep": rep
            })

index = int(sys.argv[1])
size = int(sys.argv[2])
exp = experiments[index % len(experiments)]
print(exp)
algoIndex = algorithms.index(exp["algo"])
algo = exp["algo"]
prob = exp["prob"]
rep = exp["rep"]
index = int(index / len(experiments))

folder = "output/{}_{}_{}_{}".format(algo, prob, rep, index)
if not os.path.exists(folder):
    os.mkdir(folder)
output = open(os.path.join(folder, "output.csv"), "w")
if algoIndex < oa_start:
    output.write("Index, ResultFound, time, score, ResultDepth, MaxDepth, Iterations\n")
else:
    output.write("Index, ResultFound, time, score, Generations\n")
output.close()
fitness1Folder = "{}/fitness1".format(folder)
if not os.path.exists(fitness1Folder):
    os.mkdir(fitness1Folder)
fitnessLess1Folder = "{}/fitnessLess1".format(folder)
if not os.path.exists(fitnessLess1Folder):
    os.mkdir(fitnessLess1Folder)

for i in range(size):
    if algoIndex < oa_start:
        env = PcgrlEnv(prob, rep)
        runner = algorithms_dict[algo](env)
        if algo == "MCTS":
            runner.run(env, 60, c_value[prob], roll_value[prob])
        else:
            runner.run(env, 60)

        total_time = int(runner.time_out * 1000)
        output = open(os.path.join(folder, "output.csv"), "a")
        output.write("{}, {}, {}, {}, {}, {}, {}\n".format(i, runner.best_node.win, total_time, runner.best_node.heuristic, runner.best_node.depth, runner.deep_node.depth, runner.checked_nodes))
        output.close()
        if runner.best_node.win:
            env.set_observation(runner.best_node.obs)
            image = env._prob.render(env._rep._map)
            image.save("{}/fitness1/{}.png".format(folder, i), "PNG")
            f = open("{}/fitness1/{}.txt".format(folder, i), "w")
            f.write(str(runner.best_node))
            f.close()
        else:
            env.set_observation(runner.best_node.obs)
            image = env._prob.render(env._rep._map)
            image.save("{}/fitnessLess1/{}.png".format(folder, i), "PNG")
            f = open("{}/fitnessLess1/{}.txt".format(folder, i), "w")
            f.write(str(runner.best_node))
            f.close()
    else:
        env = PcgrlEnv(prob, rep)
        runner = algorithms_dict[algo](env)
        runner.run(env)

        total_time = int(runner.time_out * 1000)
        output = open(os.path.join(folder, "output.csv"), "a")
        output.write("{}, {}, {}, {}, {}\n".format(i, runner.get_best().win, total_time, runner.get_best().fitness, runner.gen))
        output.close()
        if runner.get_best().win:
            env.set_observation(runner.get_best().obs)
            image = env._prob.render(env._rep._map)
            image.save("{}/fitness1/{}.png".format(folder, i), "PNG")
            f = open("{}/fitness1/{}.txt".format(folder, i), "w")
            f.write(str(runner.get_best()))
            f.close()
        else:
            env.set_observation(runner.get_best().obs)
            image = env._prob.render(env._rep._map)
            image.save("{}/fitnessLess1/{}.png".format(folder, i), "PNG")
            f = open("{}/fitnessLess1/{}.txt".format(folder, i), "w")
            f.write(str(runner.get_best()))
            f.close()
