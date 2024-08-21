
import ast
import numpy as np
import pandas as pd
from seeds import known_seeds
from utils import save_solution
from utils import load_problem_data
from evaluation import get_actual_demand

from helpers.decision_maker import DecisionMaker
from helpers.datacenters import Datacenter
from helpers.server_type import Server

def get_my_solution(d):
    _, datacenters, servers, selling_prices = load_problem_data()
    
    decision_maker = DecisionMaker()
    
    s = {s.server_generation: Server(s.server_generation,ast.literal_eval(s.release_time),s.purchase_price, 
                                          s.slots_size, s.energy_consumption,s.capacity,s.life_expectancy,
                                          s.cost_of_moving,s.average_maintenance_fee) for s in servers.itertuples()}
    
    decision_maker.server_types = s
    decision_maker.datacenters = {dc.datacenter_id: Datacenter(dc.datacenter_id, dc.cost_of_energy, dc.latency_sensitivity, 
                                              dc.slots_capacity, s) for dc in datacenters.itertuples()}
    
    # This is just a placeholder.
    return [{}]


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

