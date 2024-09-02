import numpy as np
import pandas as pd
import pulp
from evaluation import get_actual_demand, get_known
#from helpers.decision_maker import DecisionMaker
from helpers.datacenters import Datacenter
from helpers.server_type import Server
from seeds import known_seeds
from utils import load_problem_data

class moveLP:


    def __init__(self, datacenters: dict[str, Datacenter], server_types: dict[str, Server], demand, timestep: int):
        self.datacenters = datacenters
        self.server_types = server_types
        self.demand  = get_demand(demand, timestep)
        self.timestep = timestep

        ## create model 
        self.model = pulp.LpProblem("moves", pulp.LpMaximize)
        self.solver = pulp.PULP_CBC_CMD(msg=0)


        self.createVariables()
        self.createConstraints()
        self.createObjective()



        




        ## CONSTRAINTS 


        

        


        

    def createVariables(self):

        ## create decision variables 
        self.variables = pulp.LpVariable.dicts(     name = "move",
                                                    indices = [
                                                        (dc_from.name+"_"+dc_to.name+"_"+server.name)
                                                            for dc_from in self.datacenters.values()
                                                                for dc_to in self.datacenters.values()
                                                                    for server in self.server_types.values()
                                                            if dc_from != dc_to
                                                        ],
                                                    
                                                    cat = "Integer"
                                            )
        self.addVariables = pulp.LpVariable.dicts( 
            name = "add",
            indices=[
                (dc.name+"_"+server.name) 
                for dc in self.datacenters.values()
                for server in self.server_types.values()
                if server.canBeDeployed(self.timestep)
            ],

            cat="Integer"
        )

        self.removeVariables = pulp.LpVariable.dicts( 
            name = "remove",
            indices=[
                (dc.name+"_"+server.name) 
                for dc in self.datacenters.values()
                for server in self.server_types.values()
            ],

            cat="Integer"
        )
        
        ## set bounds for variables
        for var in self.variables:

            self.variables[var].lowBound = 0

            ## gets details about var ...  var[0] = dc_from, var[1]=dc_to , var[2] = server
            var_details = var.split("_")
            self.variables[var].upBound =   self.datacenters[ var_details[0] ].getServerStock( var.split("_")[2] )

        for var in self.addVariables:
            self.addVariables[var].lowBound=0
            var_details = var.split("_")
            latency = self.datacenters[var_details[0]].latency_sensitivity
            self.addVariables[var].upBound= self.demand[latency][var_details[1]]

        for var in self.removeVariables:
            self.removeVariables[var].lowBound=0
            var_details = var.split("_")
            self.removeVariables[var].upBound = self.datacenters[var_details[0]].getServerStock(var_details[1])    
        


    def createConstraints(self):
         
        ## the sum of all the servers moving from f should be less than the number of servers at f  
        for s in self.server_types:
            for f,dc in self.datacenters.items():

                self.model += (pulp.lpSum( [ self.variables[var] for var in self.variables if var.split("_")[0]==f  and var.split("_")[2]==s] 
                                          + [self.removeVariables[var] for var in self.removeVariables if var.split("_")[0]==f and var.split("_")[1]==s]  ) <= dc.getServerStock(s),
                                f+s+" Move + Add Less Than Current Stock Constraint")
            
        for s in self.server_types:
            for f,dc in self.datacenters.items():

                self.model += (pulp.lpSum( [ self.variables[var] for var in self.variables if var.split("_")[0]==f  and var.split("_")[2]==s] ) <= dc.getServerStock(s),
                               f+s+" Move Less Than Current Stock Constraint")

        ## the number of servers of a particular type moving to a datacentre, shouldnt exceed the demand for that server at that datacenter
        for s in self.server_types:            
            for latency in get_known('latency_sensitivity'):

                self.model += (
                                pulp.lpSum(  [self.variables[var] * self.server_types[var.split("_")[2]].capacity for var in self.variables if self.datacenters[var.split("_")[1]].latency_sensitivity==latency and var.split("_")[2]==s ] 
                                            + [self.addVariables[var] * self.server_types[var.split("_")[1]].capacity for var in self.addVariables  if self.datacenters[var.split("_")[0]].latency_sensitivity==latency and var.split("_")[1]==s ]
                                              + [-self.removeVariables[var] * self.server_types[var.split("_")[1]].capacity for var in self.removeVariables  if self.datacenters[var.split("_")[0]].latency_sensitivity==latency and var.split("_")[1]==s ]   ) 
                                    <=
                                    self.demand[latency][s] - sum( [ 
                                                                    dc.getServerStock(s)*self.server_types[s].capacity
                                                                    for dc in self.datacenters.values()
                                                                    if dc.latency_sensitivity==latency 
                                                                ])
                                , s+latency+ " Adds - Removes + Moves to datacenters of this latency shouldnt exceed the demand Constraint"
                                )
                

        ## the total slot size of the servers going in and out of a datacenter should not exceed the slot capacity left
        for t, dc in self.datacenters.items():
                
                self.model += (
                                    pulp.lpSum(
                                         [  
                                              (self.variables[var] * self.server_types[var.split("_")[2]].slots_size )
                                                for var in self.variables
                                                if var.split("_")[1]==t
                                        ]
                                        +
                                        [
                                             (-self.variables[var] * self.server_types[ var.split("_")[2]].slots_size )
                                                for var in self.variables
                                                if var.split("_")[0]==t
                                        ]
                                        +
                                        [
                                            (self.addVariables[var] * self.server_types[var.split("_")[1]].slots_size)
                                                for var in self.addVariables
                                                if var.split("_")[0]==t
                                        ]
                                        +
                                        [
                                            (-self.removeVariables[var] * self.server_types[var.split("_")[1]].slots_size)
                                                for var in self.removeVariables
                                                if var.split("_")[0]==t
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
        self.model += pulp.lpSum(
            [
                self.getAddObjectiveCoeff(var) * self.addVariables[var] for var in self.addVariables
            ]
            +
            [
                self.getMoveObjectiveCoeff(var) * self.variables[var] for var in self.variables
            ]
            +
            [
                self.getRemoveObjectiveCoeff(var) * self.removeVariables[var] for var in self.removeVariables
            ]
        )

    def getAddObjectiveCoeff(self, var):
        var_details = var.split("_")
        
        datacenter = self.datacenters[var_details[0]]
        latency = datacenter.latency_sensitivity
        server = self.server_types[var_details[1]]
        profit = server.selling_prices[latency] * server.capacity - ( (server.energy_consumption * datacenter.cost_of_energy)
                                                                     + (server.purchase_price / server.life_expectancy)
                                                                     + (server.avg_maintenance_fee / server.life_expectancy))
        return profit
    
    def getMoveObjectiveCoeff(self, var):
        var_details = var.split("_")
        
        datacenter = self.datacenters[var_details[1]]
        latency = datacenter.latency_sensitivity
        server = self.server_types[var_details[2]]
        profit = server.selling_prices[latency] * server.capacity - ( (server.energy_consumption * datacenter.cost_of_energy)
                                                                     + (server.cost_of_moving )
                                                                     + (server.avg_maintenance_fee ))
        return profit

    def getRemoveObjectiveCoeff(self, var):
        var_details = var.split("_")
        
        datacenter = self.datacenters[var_details[0]]
        latency = datacenter.latency_sensitivity
        server = self.server_types[var_details[1]]
        profit = -(server.selling_prices[latency] * server.capacity) + ( (server.energy_consumption * datacenter.cost_of_energy)
                                                                        + (server.avg_maintenance_fee / server.life_expectancy))
        return profit
    
    def solve(self):
        self.model.solve(self.solver)
    
# demand = pd.read_csv('./data/demand.csv')

# np.random.seed(0)

#     # GET THE DEMAND
# actual_demand = get_actual_demand(demand)

 
def get_demand(demand, timestep: int):
    demands = dict()
    for latency in get_known('latency_sensitivity'):
        ls_demand = dict()
        ts_demand = demand.loc[demand['time_step']==timestep].copy()
        for server in get_known('server_generation'):
            s_demand = ts_demand.loc[ts_demand['server_generation']==server].copy()
            if s_demand.empty:
                ls_demand[server] = 0
            else:
                ls_demand[server] = s_demand.iloc[0][latency]
        demands[latency] = ls_demand
    return demands


# _, datacenters, servers, selling_prices = load_problem_data()
    
# decision_maker = DecisionMaker(datacenters,servers,selling_prices, actual_demand)

# m = moveLP(decision_maker.datacenters,
#        decision_maker.server_types,
#        get_demand(1),
#        1
#        )
# print(m.solve())
# for var in m.removeVariables:
#     print(var + ": " + str(m.removeVariables[var].varValue))
# for var in m.variables:
#     print(var + ": " + str(m.variables[var].varValue))