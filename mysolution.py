
import math
import numpy as np
import pandas as pd
from seeds import known_seeds
from utils import save_solution
from utils import load_problem_data
from evaluation import get_known
from evaluation import get_actual_demand

from helpers.decision_maker import DecisionMaker

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
                if s.split('_')[1] != latency_sensitivity:
                    continue
                server_demand_df = ts_demand.loc[(ts_demand['server_generation']==s.split('_')[0])].copy()
                if server_demand_df.empty:
                    server_demands[s] = 0
                else:
                    server_demands[s] = server_demand_df.iloc[0][latency_sensitivity]
            
            for dc in dcs.keys():
                coeffs = decision_maker.getDemandCoeffs(dcs)
                demands_list = decision_maker.extractRelevantData(dc,server_demands, latency_sensitivity
                                                                                              ,coeffs[dc])
                to_add, to_remove = decision_maker.getAddRemove(demands_list, dc, latency_sensitivity)
                
                
                for serv_remove in to_remove.keys():
                    if math.ceil(to_remove[serv_remove]) == 0:
                        continue
                    decision_maker.sellServers(dc,serv_remove,math.ceil(to_remove[serv_remove]))

                for serv_add in to_add.keys():
                    if int(to_add[serv_add]) == 0:
                        continue
                    decision_maker.buyServers(dc,serv_add,int(to_add[serv_add]))
                
                
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

