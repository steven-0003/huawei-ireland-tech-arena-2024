
import numpy as np
import pandas as pd
from seeds import known_seeds
from utils import save_solution
from evaluation import get_actual_demand
from evaluation import get_known

from stockpyl.supply_chain_network import SupplyChainNetwork
from stockpyl.supply_chain_node import SupplyChainNode
from stockpyl.supply_chain_product import SupplyChainProduct
from stockpyl.demand_source import DemandSource
from stockpyl.policy import Policy
from stockpyl.sim import simulation
from stockpyl.sim_io import write_results

import statistics


def get_my_solution(d):
    #Create networks for each latency sensitivity
    high_network = SupplyChainNetwork()
    medium_network = SupplyChainNetwork()
    low_network = SupplyChainNetwork()

    # Represent each datacenter as a node
    dc1 = SupplyChainNode(index=1, name="DC1", supply_type='U', local_holding_cost=0.25)
    dc2 = SupplyChainNode(index=2, name="DC2", supply_type='U', local_holding_cost=0.35)
    dc3 = SupplyChainNode(index=3, name="DC3", supply_type='U', local_holding_cost=0.65)
    dc4 = SupplyChainNode(index=4, name="DC4", supply_type='U', local_holding_cost=0.75)

    dc1.inventory_policy = {}
    dc2.inventory_policy = {}
    dc3.inventory_policy = {}
    dc4.inventory_policy = {}

    low_network.add_node(dc1)
    medium_network.add_node(dc2)
    high_network.add_node(dc3)
    high_network.add_successor(dc3,dc4)

    #Get the nodes (datacenters) from each network for easier access
    h_nodes =  {n.index: n for n in high_network.nodes}
    m_nodes = {n.index: n for n in medium_network.nodes}
    l_nodes = {n.index: n for n in low_network.nodes}

    products = {}
    i = 1 #product id
    
    #get demand for each server and timestamp
    servers = get_known('server_generation') 
    for server in servers:
        high_demand = []
        medium_demand = []
        low_demand = []
        for ts in range(1, get_known('time_steps')+1):
            server_df = actual_demand.loc[(actual_demand['time_step']==ts) 
                                          & (actual_demand['server_generation']==server)].copy()
            
            #There is no demand of this particular server at the current timestamp
            if server_df.empty:
                high_demand.append(0)
                medium_demand.append(0)
                low_demand.append(0)
            else:
                high_demand.append(server_df.iloc[0]['high'])
                medium_demand.append(server_df.iloc[0]['medium'])
                low_demand.append(server_df.iloc[0]['low'])

        #Add all servers to the datacenters of each latency sensitivity
        
        products[i] = SupplyChainProduct(index=i, name=server+"_h")
        #products[i].demand_source = DemandSource(type='D', demand_list=high_demand)
        for h in h_nodes.keys():
            h_nodes[h].add_product(products[i])
        for h in h_nodes.keys():
            h_nodes[h].inventory_policy[i] = Policy(type='rQ', order_quantity=statistics.mean(high_demand),
                                                    reorder_point=80, node=h_nodes[h], product=products[i])
        i+=1

        products[i] = SupplyChainProduct(index=i, name=server+"_m")
        #products[i].demand_source = DemandSource(type='D', demand_list=medium_demand)
        for m in m_nodes.keys():
            m_nodes[m].add_product(products[i])
        for m in m_nodes.keys():
            m_nodes[m].inventory_policy[i] = Policy(type='rQ', order_quantity=statistics.mean(medium_demand), 
                                                    reorder_point=80, node=m_nodes[m], product=products[i])
        i+=1

        products[i] = SupplyChainProduct(index=i, name=server+"_l")
        #products[i].demand_source = DemandSource(type='D', demand_list=low_demand)
        for l in l_nodes.keys():
            l_nodes[l].add_product(products[i])
        for l in l_nodes.keys():
            l_nodes[l].inventory_policy[i] = Policy(type='rQ', order_quantity=statistics.mean(low_demand),
                                                    reorder_point=80, node=l_nodes[l], product=products[i])
        low_network.add_product(products[i])
        i+=1

    #Simulate and produce the results for all networks
    
    simulation(network=high_network, num_periods=get_known('time_steps'))
    write_results(high_network, num_periods=get_known('time_steps'), 
                  columns_to_print=['basic', 'costs', 'RM', 'ITHC'], write_csv=True, csv_filename='high_res.csv')
    
    simulation(network=medium_network, num_periods=get_known('time_steps'))
    write_results(medium_network, num_periods=get_known('time_steps'), 
                  columns_to_print=['basic', 'costs', 'RM', 'ITHC'], write_csv=True, csv_filename='med_res.csv')
    
    simulation(network=low_network, num_periods=get_known('time_steps'))
    write_results(low_network, num_periods=get_known('time_steps'), 
                  columns_to_print=['basic', 'costs', 'RM', 'ITHC'], write_csv=True, csv_filename='low_res.csv')

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

