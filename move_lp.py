import numpy as np
import pandas as pd
import pulp
from evaluation import get_actual_demand
from helpers.decision_maker import DecisionMaker
from seeds import known_seeds
from utils import load_problem_data

class moveLP:


    def __init__(self, datacenters,server_types, demand, predicted_demand):
        self.datacenters = datacenters
        self.server_types = server_types
        self.demand  = demand
        self.predicted_demand = predicted_demand



        ## create model 
        self.model = pulp.LpProblem("moves", pulp.LpMaximize)


        self.createVariables()
        self.createConstraints()
        self.createObjective()



        




        ## CONSTRAINTS 


        

        print(self.variables)


        

    def createVariables(self):

        ## create decision variables 
        self.variables = pulp.LpVariable.dicts(     name = "move",
                                                    indices = [
                                                        (dc_from.name+"_"+dc_to.name+"_"+server.name)
                                                            for dc_from in self.datacenters
                                                                for dc_to in self.datacenters
                                                                    for server in self.server_types
                                                            if dc_from != dc_to and dc_from.latency_sensitivity != dc_to.latency_sensitivity
                                                        ],
                                                    
                                                    cat = "Integer"
                                            )
        
        ## set bounds for variables
        for var in self.variables.viewkeys():

            self.variables[var].lowBound = 0

            ## gets details about var ...  var[0] = dc_from, var[1]=dc_to , var[2] = server
            var_details = var.split("_")
            self.variables[var].upBound =   self.datacenters[ var_details[0] ].getServerStock( var.split("_")[-1] )          
        


    def createConstraints(self):
         
        ## the sum of all the servers moving from f should be less than the number of servers at f  
        for f,dc in self.datacenters.items():

            self.model += (pulp.lpSum( [ var for var in self.variables.viewkeys() if var.split("_")[0]==f  ]  ) <= dc.getStockLevel(),
                            f+" Move Less Than Current Stock Constraint")
            


        ## the number of servers of a particular type moving to a datacentre, shouldnt exceed the demand for that server at that datacenter
        for s in self.server_types:            
            for t,dc in self.datacenters.items():

                self.model += (
                                pulp.lpSum(  [var for var in self.variables.viewkeys() if var.split("_")[1]==t and var.split("_")[2]==s ]    ) 
                                    <=
                                    demand[t][s]
                                , t+ " Moves to this datacentre shouldnt exceed the demand Constraint"
                                )
                

        ## the total slot size of the servers going in and out of a datacenter should not exceed the slot capacity left
        for t, dc in self.datacenters.items():
                
                self.model += (
                                    pulp.lpSum(
                                         [  
                                              (var * self.server_types[var.split("_")[2]].slots_size )
                                                for var in self.variables.viewkeys() 
                                                if var.split("_")[1]==t
                                        ]
                                        +
                                        [
                                             (-var * self.server_types[ var.split("_")[1]].slots_size )
                                                for var in self.variables.viewkeys() 
                                                if var.split("_")[1]==t
                                        ]
                                    )

                                    <= 

                                    dc.slots_capacity-dc.inventory_level

                                    , t + " Slots Taken Do Not Exceed Capacity Constraint"

                            )
         
    
    def createObjective(self):
         
        # self.model +=  pulp.lpSum(
        #                             [
        #                                 (var * )
        #                             ] 
        #                         )
        pass


    


demand = pd.read_csv('./data/demand.csv')

np.random.seed(0)

    # GET THE DEMAND
actual_demand = get_actual_demand(demand)

 



_, datacenters, servers, selling_prices = load_problem_data()
    
decision_maker = DecisionMaker(datacenters,servers,selling_prices, actual_demand)

moveLP(decision_maker.datacenters.values(),
       decision_maker.server_types.values(),
       decision_maker.demand,decision_maker.demand
       )