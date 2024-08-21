from scipy.optimize import linprog


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
