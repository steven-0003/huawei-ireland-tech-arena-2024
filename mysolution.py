
import ast
import numpy as np
import pandas as pd
from seeds import known_seeds
from utils import save_solution
from utils import load_problem_data
from evaluation import get_known
from evaluation import get_time_step_demand
from evaluation import get_actual_demand

from helpers.decision_maker import DecisionMaker
from helpers.datacenters import Datacenter
from helpers.server_type import Server

def get_my_solution(d):
    _, datacenters, servers, selling_prices = load_problem_data()
    
    decision_maker = DecisionMaker(datacenters,servers,selling_prices)
    
    # s = {s.server_generation: Server(s.server_generation,ast.literal_eval(s.release_time),s.purchase_price, 
    #                                       s.slots_size, s.energy_consumption,s.capacity,s.life_expectancy,
    #                                       s.cost_of_moving,s.average_maintenance_fee) for s in servers.itertuples()}
    
    # decision_maker.server_types = s
    # decision_maker.datacenters = {dc.datacenter_id: Datacenter(dc.datacenter_id, dc.cost_of_energy, dc.latency_sensitivity, 
    #                                           dc.slots_capacity, s) for dc in datacenters.itertuples()}

    for time_step in range(1, get_known('time_steps')+1):
        decision_maker.step()
        for latency_sensitivity in get_known('latency_sensitivity'):
            dcs = decision_maker.getLatencyDataCenters(latency_sensitivity)
            
            ts_demand = d.loc[(d['time_step']==time_step)].copy()

            server_demands = {}
            for s in decision_maker.server_types.keys():
                print(s)
                if s.split('_')[1] != latency_sensitivity:
                    continue
                server_demand_df = ts_demand.loc[(ts_demand['server_generation']==s.split('_')[0])].copy()
                if server_demand_df.empty:
                    server_demand_df[s] = 0
                else:
                    server_demands[s] = server_demand_df.iloc[0][latency_sensitivity]
            
            for dc in dcs.keys():
                print(server_demands)
                server_sizes, demands_list, server_stock = decision_maker.extractRelevantData(dc,server_demands)
                print(demands_list)
                
    return decision_maker.solution


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

