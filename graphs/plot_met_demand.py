import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import (load_problem_data,
                   load_solution)

from seeds import known_seeds
from helpers.decision_maker import DecisionMaker
from evaluation import get_known

# Load the demand.csv file
demand_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'demand.csv'))
demand = pd.read_csv(demand_path)

def plot_demand_vs_predicted_vs_actual(seed, decision_maker: DecisionMaker):
    np.random.seed(seed)
    
    cpu_met_demand = {}
    gpu_met_demand = {}

    # Loop over time steps and capture demand and met demand
    for timestep in range(1, get_known('time_steps') + 1):
        decision_maker.timestep = timestep
        current_demand = decision_maker.processDemand()  # Call processDemand from the decision_maker instance
        decision_maker.checkConstraints()
        decision_maker.step()

        for latency in get_known('latency_sensitivity'):
            for server in decision_maker.server_types.keys():
                met_demand = sum([
                    dc.inventory[server].count for dc in decision_maker.datacenters.values()
                    if dc.latency_sensitivity == latency
                ])
                total_demand = current_demand[latency][server]

                if server.startswith('CPU'):
                    if server not in cpu_met_demand:
                        cpu_met_demand[server] = []
                    cpu_met_demand[server].append((timestep, total_demand, met_demand))
                elif server.startswith('GPU'):
                    if server not in gpu_met_demand:
                        gpu_met_demand[server] = []
                    gpu_met_demand[server].append((timestep, total_demand, met_demand))

    # Plotting the demand
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Plot CPU demand
    for cpu, demand_data in cpu_met_demand.items():
        time_steps, total_demand, met_demand = zip(*demand_data)
        ax1.plot(time_steps, total_demand, label=f'{cpu} Total Demand', linestyle='--')
        ax1.plot(time_steps, met_demand, label=f'{cpu} Met Demand')

    ax1.set_title(f'CPU Demand vs. Met Demand (Seed {seed})')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Demand')
    ax1.grid(True)
    ax1.legend()

    # Plot GPU demand
    for gpu, demand_data in gpu_met_demand.items():
        time_steps, total_demand, met_demand = zip(*demand_data)
        ax2.plot(time_steps, total_demand, label=f'{gpu} Total Demand', linestyle='--')
        ax2.plot(time_steps, met_demand, label=f'{gpu} Met Demand')

    ax2.set_title(f'GPU Demand vs. Met Demand (Seed {seed})')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Demand')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'./graphs/met_demand_plot_seed_{seed}.png')
    plt.show()
    plt.close()

# Loop through all seeds and generate plots
seeds = known_seeds('training')

for seed in seeds:

      # LOAD SOLUTION
    solution = load_solution(f'./output/{seed}.json')

    # LOAD PROBLEM DATA
    demand, datacenters, servers, selling_prices = load_problem_data()


    dm = DecisionMaker(datacenters, servers, selling_prices, demand)
    plot_demand_vs_predicted_vs_actual(seed, dm)