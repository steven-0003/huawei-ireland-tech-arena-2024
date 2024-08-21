import numpy as np
import pandas as pd
from collections import deque
from seeds import known_seeds
from utils import save_solution
from evaluation import get_actual_demand

def initialise_queues_and_counts(data_centers, server_types): # Initialise the queues and server counts
    queues = {dc: {stype: deque() for stype in server_types} for dc in data_centers}
    server_counts = {dc: {stype: 0 for stype in server_types} for dc in data_centers}
    return queues, server_counts

def add_server_to_queue(queues, server_counts, datacenter_id, server_type, server_id, timestamp): # Add server to queue and update server_counts
    queues[datacenter_id][server_type].append((server_id, timestamp))
    server_counts[datacenter_id][server_type] += 1 #increase the server count

def evict_servers_from_queue(queues, server_counts, datacenter_id, server_type, num_servers): # Evict servers from the queue and update server_counts
    evicted_servers = []
    for i in range(num_servers):
        if queues[datacenter_id][server_type]:
            server = queues[datacenter_id][server_type].popleft() #idk to pop from the right or left
            evicted_servers.append(server)
            server_counts[datacenter_id][server_type] -= 1 #decrease the server count
    return evicted_servers

# def move_servers_between_queues(queues, server_counts, source_dc, target_dc, server_type, num_servers): idk how to do this

def get_my_solution(demand):
    data_centers = demand['datacenter_id'].unique()
    server_types = ['CPU', 'GPU']
    
    # Initialize the queues and server counts
    queues, server_counts = initialise_queues_and_counts(data_centers, server_types)
    
    solution = []
    
    # demand, datacenters, servers, selling_prices = load_problem_data()

    for time_stamp, row in demand.iterrows():
        datacenter_id = row['datacenter_id']
        server_generation = row['server_generation']
        server_id = f'{server_generation}_{time_stamp}'
        server_type = server_generation.split('.')[0]
        
        # Add server to the queue when buying
        add_server_to_queue(queues, server_counts, datacenter_id, server_type, server_id, time_stamp)
        
        #FIXME: Example conditions to evict servers from the queue, put it inside some condtion
        if server_counts[datacenter_id]['CPU'] > 5: #example
            evict_servers_from_queue(queues, server_counts, datacenter_id, 'CPU', 1) #same can be done for GPU
        
        # Append the action to the solution
        solution.append({
            "time_stamp": time_stamp,
            "datacenter_id": datacenter_id,
            "server_generation": server_generation,
            "server_id": server_id,
            # "action": decisionMaker() #FIXME: Add the action here, it shuold use a decision maker function.
        })
    
    return solution


# def decisionMaker():
    # makes decision on what action to take based on conditions


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