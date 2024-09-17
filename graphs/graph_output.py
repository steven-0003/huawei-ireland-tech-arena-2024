#graph_output.py
import sys
import os

from matplotlib import pyplot as plt
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import (load_problem_data,
                   load_solution)
# from evaluation import evaluation_function
# from lifetime_graph import evaluation_function, getTimestepsList, getLifespanList
from profit_graph import evaluation_function, getTimestepsList, getProfitList
# from utilisation_graph import evaluation_function
from seeds import known_seeds
import statistics



seeds = known_seeds('test')

scores = []
for seed in seeds:
    # LOAD SOLUTION
    solution = load_solution(f'./output/{seed}.json')

    # LOAD PROBLEM DATA
    demand, datacenters, servers, selling_prices = load_problem_data()

    graph = evaluation_function(solution,
                                demand,
                                datacenters,
                                servers,
                                selling_prices,
                                seed=seed,
                                verbose=0)

    # timesteps_list = getTimestepsList()
    # lifespan_list = getLifespanList()

    # print(f'seeds: {seeds}')
    # print(f'generating graph with seed {seed}')

    # plt.figure(figsize=(10, 6))
    # plt.plot(timesteps_list, lifespan_list, marker='o', linestyle='-', color='b')
    # plt.xlabel('Time-Step')
    # plt.ylabel(f'lifetime for seed {seed}')
    # plt.title(f'lifetime vs timestep for seed {seed}')
    # plt.tight_layout()
    # plt.grid(True)
    # plt.savefig(f"./graphs/lifetime_{seed}.png")
    # plt.close()


    timesteps_list = getTimestepsList()
    # profit_list = getProfitList()
    profit_list = np.cumsum(getProfitList())


    print(f'seeds: {seeds}')
    print(f'generating graph with seed {seed}')

    plt.figure(figsize=(10, 6))
    plt.plot(timesteps_list, profit_list, marker='o', linestyle='-', color='b')
    plt.xlabel('Timestep')
    plt.ylabel(f'Profit for seed {seed}')
    plt.title(f'Profit vs timestep for seed {seed}')
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"./graphs/profit_{seed}.png")
    plt.close()
