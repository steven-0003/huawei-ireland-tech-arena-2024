import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from seeds import known_seeds
from helpers.decision_maker import DecisionMaker
from evaluation import get_known
from evaluation import get_actual_demand
from helpers.datacenters import Datacenter

# Add road_to_dublin to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def known_seeds(mode):
    if mode == 'training':
        return [1741, 3163, 6053, 2237, 8237, 8933, 4799, 1061, 2543, 8501]
    elif mode == 'test':
        return []
    return []

# Load the demand.csv file
demand_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','road-to-dublin', 'data', 'demand.csv'))
demand = pd.read_csv(demand_path)



def calculate_met_demand(decision_maker: DecisionMaker, timestep: int) -> dict[str, dict[str, float]]:
    """
    Calculate the met demand for each latency sensitivity and server type at a given timestep.
    Returns a nested dictionary where:
    - First key is the latency sensitivity
    - Second key is the server type
    - Value is the total met demand for that latency and server type at the current timestep.
    """
    met_demand = {}
    current_demand = decision_maker.processDemand()  # Get the actual demand for the current timestep

    for latency_sensitivity in decision_maker.get_known('latency_sensitivity'):
        # Initialise the dictionary for this latency
        met_demand[latency_sensitivity] = {}

        # Get all datacenters with the current latency sensitivity
        datacenters_with_latency = decision_maker.getLatencyDataCenters(latency_sensitivity)
        
        for server_type in decision_maker.server_types.keys():
            total_met_demand = 0
            
            # Loop over datacenters to calculate the number of deployed servers
            for dc_name, datacenter in datacenters_with_latency.items():
                deployed_servers = len(datacenter.inventory.get(server_type, []))
                server_capacity = decision_maker.server_types[server_type].capacity
                total_met_demand += deployed_servers * server_capacity
            
            # Store the met demand for this server type
            met_demand[latency_sensitivity][server_type] = total_met_demand

    return met_demand


def plot_demand_for_seed(seed: int, decision_maker: DecisionMaker) -> None:
    """
    Plot the actual demand vs met demand over time for each latency sensitivity and server type.
    """
    # Set the seed for reproducibility
    np.random.seed(seed)
    
    timesteps = range(1, get_known('time_steps') + 1)
    latency_sensitivities = get_known('latency_sensitivity')

    # Initialise data structures to store demand over time
    demand_data = {latency: {server: [] for server in decision_maker.server_types.keys()} for latency in latency_sensitivities}
    met_demand_data = {latency: {server: [] for server in decision_maker.server_types.keys()} for latency in latency_sensitivities}
    current_demand  = get_actual_demand(demand)
    # Loop over each timestep and collect demand and met demand
    for timestep in timesteps:
        # Process the demand for the current timestep
        decision_maker.step()
        current_met_demand = calculate_met_demand(decision_maker, timestep)

        # Store data for each latency and server type
        for latency in latency_sensitivities:
            for server in decision_maker.server_types.keys():
                demand_data[latency][server].append(current_demand[latency].get(server, 0))
                met_demand_data[latency][server].append(current_met_demand[latency].get(server, 0))

    # Plot demand vs met demand for each latency and server
    for latency in latency_sensitivities:
        plt.figure(figsize=(10, 6))
        for server in decision_maker.server_types.keys():
            plt.plot(timesteps, demand_data[latency][server], label=f'Demand - {server}', linestyle='--')
            plt.plot(timesteps, met_demand_data[latency][server], label=f'Met Demand - {server}')
        
        plt.title(f'Demand vs Met Demand for Latency: {latency}')
        plt.xlabel('Timestep')
        plt.ylabel('Demand')
        plt.legend()
        plt.grid(True)
        plt.show()


# Code to loop through all seeds and generate plots
seeds = known_seeds('training')

for seed in seeds:
    plot_demand_for_seed(seed, demand)