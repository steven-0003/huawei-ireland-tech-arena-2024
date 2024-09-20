from utils import (load_problem_data,
                   load_solution)
from evaluation import evaluation_function
from seeds import known_seeds
import statistics

seeds = known_seeds()

scores = []



for seed in seeds:

    
    print(f"SEED: #{seed}")
    

    # LOAD SOLUTION
    fleet, pricing_strategy = load_solution(f'./output/{seed}.json')

    # LOAD PROBLEM DATA
    demand, datacenters, servers, selling_prices, elasticity = load_problem_data()

    # EVALUATE THE SOLUTION
    score = evaluation_function(fleet,
                                pricing_strategy,
                                demand,
                                datacenters,
                                servers,
                                selling_prices,
                                elasticity,
                                seed=seed,
                                verbose=1)
    scores.append(score)
    print(f'Solution score: {score}')

print(f'Average solution score: {statistics.mean(scores)}')
