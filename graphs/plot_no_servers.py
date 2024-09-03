import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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

def calculate_server_changes(demand):
    # Get the actual demand using get_actual_demand function
    actual_demand = get_actual_demand(demand)
    
    # Calculate the number of servers required at each timestep
    server_counts = actual_demand.groupby('time_step').size()

    # Calculate changes in the number of servers
    server_changes = server_counts.diff().fillna(0).astype(int)
    
    return server_changes

def plot_server_changes_for_seed(seed, demand):
    # Set the random seed
    np.random.seed(seed)
    
    # Calculate the server changes
    server_changes = calculate_server_changes(demand)

    # Plot the changes in server numbers
    plt.figure(figsize=(14, 8))
    plt.plot(server_changes.index, server_changes, marker='o', linestyle='-', color='b')
    
    plt.title(f'Number of Servers Added/Removed Over Time (Seed {seed})')
    plt.xlabel('Time Step')
    plt.ylabel('Number of Servers Added/Removed')
    plt.grid(True)
    
    # Reduce the number of x-axis labels using MaxNLocator
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=20))  # Adjust nbins for more/less labels
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'./server_changes_plot_seed_{seed}.png')
    plt.show()
    plt.close()

# Main code to loop through all seeds and generate plots
seeds = known_seeds('training')

for seed in seeds:
    plot_server_changes_for_seed(seed, demand)
