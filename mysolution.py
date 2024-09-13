
import math
import numpy as np
import pandas as pd
from seeds import known_seeds
from utils import save_solution
from utils import load_problem_data
from evaluation import get_known
from evaluation import get_actual_demand

from helpers.decision_maker import DecisionMaker

def get_my_solution(d, s):
    _, datacenters, servers, selling_prices = load_problem_data()
    
    decision_maker = DecisionMaker(datacenters,servers,selling_prices, d, s)

    return decision_maker.solve()
    
  





seeds = known_seeds('test')



demand = pd.read_csv('./data/demand.csv')
for seed in seeds:
    print(f"SEED: #{seed}")
    # SET THE RANDOM SEED
    np.random.seed(seed)

    # GET THE DEMAND
    actual_demand = get_actual_demand(demand)

    # CALL YOUR APPROACH HERE
    solution = get_my_solution(actual_demand, seed)

    # SAVE YOUR SOLUTION
    save_solution(solution, f'./output/{seed}.json')

