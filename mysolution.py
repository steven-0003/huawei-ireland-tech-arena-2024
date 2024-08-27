
import math
import numpy as np
import pandas as pd
from seeds import known_seeds
from utils import save_solution
from utils import load_problem_data
from evaluation import get_known
from evaluation import get_actual_demand

from helpers.decision_maker import DecisionMaker

from statsmodels.tsa.api import Holt
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

def get_my_solution(d):
    _, datacenters, servers, selling_prices = load_problem_data()

    for server in get_known('server_generation'):
        server_demand = d.loc[(d['server_generation'] == server)].copy()
        for latency in get_known('latency_sensitivity'):
            latency_demand = server_demand[['time_step', latency]].copy()
            fit = Holt(latency_demand[latency].to_numpy(), damped_trend=True, initialization_method="estimated").fit(
                            smoothing_level=0.1, smoothing_trend=0.1
                        )
            holt_demands = fit.fittedvalues
            holt_demands[holt_demands < 0] = 0
            d.loc[d['server_generation'] == server, latency] = holt_demands.astype(int)
    
    decision_maker = DecisionMaker(datacenters,servers,selling_prices, d)

    return decision_maker.solve()
    
  





seeds = known_seeds('training')

demand = pd.read_csv('./data/demand.csv')
for seed in seeds:
    # SET THE RANDOM SEED
    np.random.seed(seed)

    # GET THE DEMAND
    actual_demand = get_actual_demand(demand)

    # CALL YOUR APPROACH HERE
    solution = get_my_solution(actual_demand)

    # SAVE YOUR SOLUTION
    save_solution(solution, f'./output/{seed}.json')

