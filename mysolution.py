
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
    _, datacenters, servers, selling_prices, _ = load_problem_data()
    
    decision_maker = DecisionMaker(datacenters,servers,selling_prices, d, s)

    fleet, prices, O = decision_maker.solve()

    return pd.DataFrame(fleet), pd.DataFrame(prices), O
    




seeds = known_seeds()

demand = pd.read_csv('./data/demand.csv')
OS = []
for seed in seeds:
    # SET THE RANDOM SEED
    np.random.seed(seed)

    # GET THE DEMAND
    actual_demand = get_actual_demand(demand)

    # CALL YOUR APPROACH HERE
    fleet, pricing_strategy, O = get_my_solution(actual_demand, seed)
    OS.append(O)

    # SAVE YOUR SOLUTION
    save_solution(fleet, pricing_strategy, f'./output/{seed}.json')

print("Average score: " + str(np.mean(OS)))