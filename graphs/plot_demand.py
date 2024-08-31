import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Add road_to_dublin to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation import get_actual_demand  

def known_seeds(mode):
    if mode == 'training':
        return [1741, 3163, 6053, 2237, 8237, 8933, 4799, 1061, 2543, 8501]
    elif mode == 'test':
        return []
    return []

# Load the demand.csv file
demand_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'demand.csv'))
demand = pd.read_csv(demand_path)

def plot_demand_for_seed(seed, demand):    
    # Set the random seed
    np.random.seed(seed)
    
    # Get the actual demand using get_actual_demand function
    actual_demand = get_actual_demand(demand)
    
    # CPU and GPU demands
    cpu_demands = {}
    gpu_demands = {}

    for cpu in [f'CPU.S{i}' for i in range(1, 5)]:
        server_demand = actual_demand[actual_demand['server_generation'] == cpu]
        total_demand = server_demand[['high', 'low', 'medium']].sum(axis=1).groupby(server_demand['time_step']).sum()
        cpu_demands[cpu] = total_demand

    for gpu in [f'GPU.S{i}' for i in range(1, 4)]:
        server_demand = actual_demand[actual_demand['server_generation'] == gpu]
        total_demand = server_demand[['high', 'low', 'medium']].sum(axis=1).groupby(server_demand['time_step']).sum()
        gpu_demands[gpu] = total_demand

    # Create subplots for CPU and GPU demand
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Plot CPU demand
    for cpu, demand in cpu_demands.items():
        ax1.plot(demand.index, demand, label=cpu)

    ax1.set_title(f'CPU Demand Over Time (Seed {seed})')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Total Demand')
    ax1.grid(True)
    ax1.legend()
    ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax1.ticklabel_format(style='plain', axis='y')  

    # Plot GPU demand
    for gpu, demand in gpu_demands.items():
        ax2.plot(demand.index, demand, label=gpu)

    ax2.set_title(f'GPU Demand Over Time (Seed {seed})')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Total Demand')
    ax2.grid(True)
    ax2.legend()
    ax2.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax2.ticklabel_format(style='plain', axis='y') 

    # Save the plots
    plt.tight_layout()
    plt.savefig(f'./graphs/demand_plot_seed_{seed}.png')
    plt.show()
    plt.close()

# Code to loop through all seeds and generate plots
seeds = known_seeds('training')

for seed in seeds:
    plot_demand_for_seed(seed, demand)