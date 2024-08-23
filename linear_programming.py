from typing import Dict, List
from typing import Tuple

from scipy.optimize import linprog

import numpy as np

from helpers.server_type import Server


def find_add_and_evict( inequality_matrix, inequality_vector, objective , decision_var_bounds ):

    ## decision_var_bounds = range that the decision variables taken 
    ## provided in this order,  [
    ##                           server1_add, server1_evict,
    ##                           server2_add, server2_evict 
    ##                           ...
    ##                          ]
    

    ## solve problem 
    results = linprog(  c=objective,
                        A_ub=inequality_matrix,
                        b_ub= inequality_vector,
                        bounds=decision_var_bounds,
                        method='highs-ds'
                    )

    ## returns optimal values for each decision variable in same order as 
    ## decision_var_bounds
     
    return results.x




# def create_inequality_matrix(server_sizes_active: List[Tuple[int,bool]] , server_capacities: List[int]  ):
def create_inequality_matrix(servers: List[Server] , actives = [bool]  ):


    assert len(servers)!=0 , "List of servers is empty" 
    assert len(servers) == len(actives) , "Length of servers does not match length of actives"
    
    active_num = len( [a for a in actives if a == True ] )

    cols = len(servers) + active_num
    rows = 1 + len(servers)


    matrix = np.zeros( (rows,cols) )


    ## size of servers to be removed
    for i in range(len(servers)):
        matrix[0,i] =  -servers[i].slots_size

    


    active_count = 0 

    for s in range(len(servers)):

        ## add capacity for servers to be removed
        matrix[1+s,s] = -servers[s].capacity

        if actives[s]: 

            ## if server active add its size to first row
            matrix[0, len(servers) + active_count] = servers[s].slots_size 

            ## if server is active add its capacity 
            matrix[1+s, len(servers)  + active_count ] = servers[s].capacity

            active_count+=1




    assert active_count == active_num



    
    return matrix



    


def create_inequality_vector(remaining_slots: int, server_demands: List[int], server_stock:List[int], server_capacities:List[int]):

    ## remaining capacity = capacity left in datacenter 
    ## server_demands = list of demand for each type of server
    ## server_stock = the current amount of each server type in the data centre

    assert len(server_demands) == len(server_stock), "Size of server demands does not match size of server stock"
    
    ## create large enough vector 
    vector = np.zeros(1+len(server_demands))


    vector[0] = remaining_slots

    ## for each server add the demand that is yet to be met 
    for i in range(len(server_demands)):

        vector[i+1] = ( server_demands[i] - (server_stock[i]*server_capacities[i])   )
    
    return vector



## need to factor in cost of energy, at data centre with energy consumption of server 
## remember scipy minimises the objective function, so need to slip all the signs
# def create_objective_vector(selling_prices: List[float] , energy_consumptions: List[float] , capacities: List[int],energy_cost: float  ):
def create_objective_vector(servers: List[Server], actives : List[bool] ,energy_cost: float  ):

    assert len(servers)==len(actives) , "Length of selling prices is not equal to the length of the energy consumptions"

    ## the fraction of a lifetime we expect a server to last, parameter we need to set 
    expected_lifetime = 0.5


    ## calculate profit for each server
    profits = [(  
                    (p.selling_price*p.capacity)
                    - 
                    (
                        (p.energy_consumption * energy_cost)
                           + 
                       (p.purchase_price / (p.life_expectancy * expected_lifetime) )
                    )
                )
                for p in servers
                ]

    
    active_num = len([i for i in actives if i==True])

    objective_vector = np.zeros( len(servers) + active_num )


    active_count = 0 
    for s in range(len(servers)):

        objective_vector[s] = -profits[s]

        ## if server is active add expected profit of adding it 
        if actives[s]:

            objective_vector[len(servers)+active_count] = profits[s]
            active_count+=1

    assert active_num == active_count
    

    return -objective_vector

    

