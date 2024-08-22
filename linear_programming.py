from typing import List

from scipy.optimize import linprog

import numpy as np


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




def create_inequality_matrix(server_sizes: List[int] ):
    ##server sizes = a listt of the slot sizes for each server    

    assert len(server_sizes)!=0 , "List of server sizes is empty" 
    
    cols = len(server_sizes) * 2
    rows = 1 + len(server_sizes)


    matrix = np.zeros( (rows,cols) )


    
    for i in range(len(server_sizes)):
        matrix[0,2*i] =  server_sizes[i]
        matrix[0,2*i +1] =  -server_sizes[i] 

    for s in range(len(server_sizes)):


        matrix[1+s,2*s] = 1
        matrix[1+s,2*s+1] = -1


        


    
    return matrix



    


def create_inequality_vector(remaining_capacity: int, server_demands: List[int], server_stock:List[int]):

    ## remaining capacity = capacity left in datacenter 
    ## server_demands = list of demand for each type of server
    ## server_stock = the current amount of each server type in the data centre

    assert len(server_demands) == len(server_stock), "Size of server demands does not match size of server stock"
    
    ## create large enough vector 
    vector = np.zeros(1+len(server_demands))


    vector[0] = remaining_capacity

    ## for each server add the demand that is yet to be met 
    for i in range(len(server_demands)):

        vector[i+1] = ( server_demands[i] - server_stock[i]   )
    
    return vector



## need to factor in cost of energy, at data centre with energy consumption of server 
## remember scipy minimises the objective function, so need to slip all the signs

def create_objective_vector(selling_prices: List[float] , energy_consumptions: List[float] , energy_cost: float  ):

    assert len(selling_prices)==len(energy_consumptions) , "Length of selling prices is not equal to the length of the energy consumptions"


    selling_prices = np.asarray(selling_prices)
    energy_consumptions = np.asarray(energy_consumptions)

    energy_consumptions *= energy_cost


    return selling_prices-energy_consumptions

    

